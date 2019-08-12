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
import llvmlite.binding as ll
from bodo.libs import hstr_ext
ll.add_symbol('mask_arr_to_bitmap', hstr_ext.mask_arr_to_bitmap)


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
    # TODO: handle or disallow reflection
    int_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    data_obj = c.pyapi.object_getattr_string(obj, "_data")
    int_arr.data = c.pyapi.to_native_value(
        types.Array(typ.dtype, 1, 'C'), data_obj).value

    mask_arr_obj = c.pyapi.object_getattr_string(obj, "_mask")
    mask_arr = c.pyapi.to_native_value(
        types.Array(types.bool_, 1, 'C'), mask_arr_obj).value

    # TODO: use this when Numba's #4435 is resolved
    # bt_typ = c.context.typing_context.resolve_value_type(mask_arr_to_bitmap)
    # bt_sig = bt_typ.get_call_type(
    #     c.context.typing_context, (types.Array(types.bool_, 1, 'C'),), {})
    # bt_impl = c.context.get_function(bt_typ, bt_sig)
    # int_arr.null_bitmap = bt_impl(c.builder, (mask_arr,))

    # XXX: workaround wrapper can be used
    # fnty = c.context.call_conv.get_function_type(bt_sig.return_type, bt_sig.args)
    # mod = c.builder.module
    # fn = lir.Function(
    #     mod, fnty, name=mod.get_unique_name('.mask_arr_conv'),
    # )
    # fn.linkage = 'internal'
    # inner_builder = lir.IRBuilder(fn.append_basic_block())
    # [inner_item] = c.context.call_conv.decode_arguments(
    #     inner_builder, bt_sig.args, fn,
    # )
    # h = bt_impl(inner_builder, (inner_item,))
    # c.context.call_conv.return_value(inner_builder, h)

    # status, retval = c.context.call_conv.call_function(
    #     c.builder, fn, bt_sig.return_type, bt_sig.args, [mask_arr],
    # )

    # int_arr.null_bitmap = retval

    n = c.pyapi.long_as_longlong(c.pyapi.call_method(obj, '__len__', ()))
    n_bytes = c.builder.udiv(c.builder.add(n,
            lir.Constant(lir.IntType(64), 7)),
            lir.Constant(lir.IntType(64), 8))
    mask_arr_struct = c.context.make_array(types.Array(types.bool_, 1, 'C'))(
        c.context, c.builder, mask_arr)
    bitmap_arr_struct = numba.targets.arrayobj._empty_nd_impl(
        c.context, c.builder, types.Array(types.bool_, 1, 'C'), [n_bytes])

    fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer(),
        lir.IntType(8).as_pointer(), lir.IntType(64)])
    fn = c.builder.module.get_or_insert_function(
        fnty, name="mask_arr_to_bitmap")
    c.builder.call(fn, [bitmap_arr_struct.data, mask_arr_struct.data, n])

    int_arr.null_bitmap = bitmap_arr_struct._getvalue()

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(int_arr._getvalue(), is_error=is_error)


@box(IntegerArrayType)
def box_int_arr(typ, val, c):
    """Box int array into pandas IntegerArray object. Null bitmap is converted
    to mask array.
    """
    # box integer array's data and bitmap
    int_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    data =  c.pyapi.from_native_value(
        types.Array(typ.dtype, 1, 'C'), int_arr.data, c.env_manager)
    bitmap_arr_data = c.context.make_array(types.Array(types.uint8, 1, 'C'))(
        c.context, c.builder, int_arr.null_bitmap).data

    # allocate mask array
    n_obj = c.pyapi.call_method(data, '__len__', ())
    n = c.pyapi.long_as_longlong(n_obj)
    mod_name = c.context.insert_const_string(c.builder.module, "numpy")
    np_class_obj = c.pyapi.import_module_noblock(mod_name)
    bool_dtype = c.pyapi.object_getattr_string(np_class_obj, 'bool_')
    mask_arr = c.pyapi.call_method(np_class_obj, 'empty', (n_obj, bool_dtype))
    mask_arr_ctypes = c.pyapi.object_getattr_string(mask_arr, 'ctypes')
    mask_arr_data = c.pyapi.object_getattr_string(mask_arr_ctypes, 'data')
    mask_arr_ptr = c.builder.inttoptr(
        c.pyapi.long_as_longlong(mask_arr_data), lir.IntType(8).as_pointer())

    # fill mask array
    with cgutils.for_range(c.builder, n) as loop:
        # (bits[i >> 3] >> (i & 0x07)) & 1
        i = loop.index
        byte_ind = c.builder.lshr(i, lir.Constant(lir.IntType(64), 3))
        byte = c.builder.load(
            cgutils.gep(c.builder, bitmap_arr_data, byte_ind))
        mask = c.builder.trunc(
            c.builder.and_(i, lir.Constant(lir.IntType(64), 7)),
            lir.IntType(8))
        val = c.builder.and_(
            c.builder.lshr(byte, mask), lir.Constant(lir.IntType(8), 1))
        # flip value since bitmap uses opposite convention
        val = c.builder.xor(val, lir.Constant(lir.IntType(8), 1))
        ptr = cgutils.gep(c.builder, mask_arr_ptr, i)
        c.builder.store(val, ptr)

    # create IntegerArray
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    arr_mod_obj = c.pyapi.object_getattr_string(pd_class_obj, 'arrays')
    res = c.pyapi.call_method(arr_mod_obj, 'IntegerArray', (data, mask_arr))

    # clean up references (TODO: check for potential refcount issues)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(n_obj)
    c.pyapi.decref(np_class_obj)
    c.pyapi.decref(bool_dtype)
    c.pyapi.decref(mask_arr_ctypes)
    c.pyapi.decref(mask_arr_data)
    c.pyapi.decref(arr_mod_obj)
    return res
