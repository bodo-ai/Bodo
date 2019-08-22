"""Nullable boolean array that stores data in Numpy format (1 byte per value)
but nulls are stored in bit arrays (1 bit per value) similar to Arrow's nulls.
Pandas converts boolean array to object when NAs are introduced.
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
from numba.array_analysis import ArrayAnalysis
from bodo.libs.str_arr_ext import kBitmask
from bodo.utils.typing import is_list_like_index_type

from llvmlite import ir as lir
import llvmlite.binding as ll
from bodo.libs import hstr_ext
ll.add_symbol('set_nulls_bool_array', hstr_ext.set_nulls_bool_array)
ll.add_symbol('is_bool_array', hstr_ext.is_bool_array)
from bodo.utils.typing import (is_overload_none, is_overload_true,
    is_overload_false, parse_dtype)


class BooleanArrayType(types.ArrayCompatible):
    def __init__(self):
        super(BooleanArrayType, self).__init__(
            name='BooleanArrayType()')

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, 'C')

    @property
    def dtype(self):
        return types.bool_

    def copy(self):
        return BooleanArrayType()


boolean_array = BooleanArrayType()


# store data and nulls as regular numpy arrays without payload machineray
# since this struct is immutable (data and null_bitmap are not assigned new
# arrays after initialization)
@register_model(BooleanArrayType)
class BooleanArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', types.Array(types.bool_, 1, 'C')),
            ('null_bitmap', types.Array(types.uint8, 1, 'C')),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(BooleanArrayType, 'data', '_data')
make_attribute_wrapper(BooleanArrayType, 'null_bitmap', '_null_bitmap')


@numba.njit
def gen_full_bitmap(data_arr):
    n = len(data_arr)
    n_bytes = (n + 7) >> 3
    return np.full(n_bytes, 255, np.uint8)


def call_func_in_unbox(func, args, arg_typs, c):
    func_typ = c.context.typing_context.resolve_value_type(func)
    func_sig = func_typ.get_call_type(
        c.context.typing_context, arg_typs, {})
    func_impl = c.context.get_function(func_typ, func_sig)

    # XXX: workaround wrapper must be used due to call convention changes
    fnty = c.context.call_conv.get_function_type(
        func_sig.return_type, func_sig.args)
    mod = c.builder.module
    fn = lir.Function(
        mod, fnty, name=mod.get_unique_name('.func_conv'),
    )
    fn.linkage = 'internal'
    inner_builder = lir.IRBuilder(fn.append_basic_block())
    inner_args = c.context.call_conv.decode_arguments(
        inner_builder, func_sig.args, fn,
    )
    h = func_impl(inner_builder, inner_args)
    c.context.call_conv.return_value(inner_builder, h)

    status, retval = c.context.call_conv.call_function(
        c.builder, fn, func_sig.return_type, func_sig.args, args,
    )
    # TODO: check status?
    return retval


@unbox(BooleanArrayType)
def unbox_bool_array(typ, obj, c):
    """
    Convert a Numpy array object to a native BooleanArray structure.
    The array's dtype can be bool or object, depending on the presense of nans.
    """
    fnty = lir.FunctionType(lir.IntType(32), [lir.IntType(8).as_pointer()])
    fn = c.builder.module.get_or_insert_function(
        fnty, name="is_bool_array")
    is_bool_dtype = c.builder.call(fn, [obj])
    # cgutils.printf(c.builder, 'is bool %d\n', is_bool_dtype)

    bool_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    cond = c.builder.icmp_unsigned('!=', is_bool_dtype, is_bool_dtype.type(0))
    with c.builder.if_else(cond) as (then, otherwise):
        with then:
            # array is bool
            bool_arr.data = c.pyapi.to_native_value(
                types.Array(types.bool_, 1, 'C'), obj).value
            bool_arr.null_bitmap = call_func_in_unbox(
                gen_full_bitmap, (bool_arr.data,),
                (types.Array(types.bool_, 1, 'C'),), c)
        with otherwise:
            # array is object
            pass

    return NativeValue(bool_arr._getvalue())


@box(BooleanArrayType)
def box_bool_arr(typ, val, c):
    """Box int array into Numpy boolean array if there are no NAs. Otherwise,
    use object array with NAs converted to np.nan.
    """
    bool_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    bool_arr_obj =  c.pyapi.from_native_value(
        types.Array(typ.dtype, 1, 'C'), bool_arr.data, c.env_manager)
    bitmap_arr_data = c.context.make_array(types.Array(types.uint8, 1, 'C'))(
        c.context, c.builder, bool_arr.null_bitmap).data

    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
        [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = c.builder.module.get_or_insert_function(
        fnty, name="set_nulls_bool_array")
    res = c.builder.call(fn, [bool_arr_obj, bitmap_arr_data])
    return res


@intrinsic
def init_bool_array(typingctx, data, null_bitmap=None):
    """Create a BooleanArray with provided data and null bitmap values.
    """
    assert data == types.Array(types.bool_, 1, 'C')
    assert null_bitmap == types.Array(types.uint8, 1, 'C')

    def codegen(context, builder, signature, args):
        data_val, bitmap_val = args
        # create bool_arr struct and store values
        bool_arr = cgutils.create_struct_proxy(
            signature.return_type)(context, builder)
        bool_arr.data = data_val
        bool_arr.null_bitmap = bitmap_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            context.nrt.incref(builder, signature.args[1], bitmap_val)

        return bool_arr._getvalue()

    sig = boolean_array(data, null_bitmap)
    return sig, codegen


# using a function for getting data to enable extending various analysis
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_bool_arr_data(A):
    return lambda A: A._data


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_bool_arr_bitmap(A):
    return lambda A: A._null_bitmap


# array analysis extension
def get_bool_arr_data_equiv(self, scope, equiv_set, args, kws):
    assert len(args) == 1 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return var, []
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_bool_arr_ext_get_bool_arr_data = \
    get_bool_arr_data_equiv


def init_bool_array_equiv(self, scope, equiv_set, args, kws):
    assert len(args) == 2 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return var, []
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_bool_arr_ext_init_bool_array = \
    init_bool_array_equiv


@overload(len)
def overload_bool_arr_len(A):
    if A == boolean_array:
        return lambda A: len(A._data)
