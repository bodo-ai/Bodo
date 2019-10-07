"""Tools for handling bodo arrays, e.g. passing to C/C++ code
"""
import numpy as np
import numba
from numba import types, cgutils
from numba.extending import (typeof_impl, type_callable, models,
    register_model, NativeValue, make_attribute_wrapper, lower_builtin, box,
    unbox, lower_getattr, intrinsic, overload_method, overload,
    overload_attribute)
import bodo
from bodo.libs.str_arr_ext import string_array_type
from bodo.utils.utils import _numba_to_c_type_map

from bodo.libs import array_tools_ext
from llvmlite import ir as lir
import llvmlite.binding as ll
ll.add_symbol('string_array_to_info', array_tools_ext.string_array_to_info)
ll.add_symbol('numpy_array_to_info', array_tools_ext.numpy_array_to_info)


class ArrayInfoType(types.Type):
    def __init__(self):
        super(ArrayInfoType, self).__init__(name='ArrayInfoType()')


array_info_type = ArrayInfoType()
register_model(ArrayInfoType)(models.OpaqueModel)


@intrinsic
def array_to_info(typingctx, arr_type):
    def codegen(context, builder, sig, args):
        in_arr, = args
        # XXX: meminfo is not updated, so array may go away if not used
        # afterwards somewhere
        # TODO: fix memory management

        # StringArray
        if arr_type == string_array_type:
            string_array = context.make_helper(
                builder, string_array_type, in_arr)
            fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                [lir.IntType(64), lir.IntType(64), lir.IntType(8).as_pointer(),
                lir.IntType(32).as_pointer(), lir.IntType(8).as_pointer()])
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="string_array_to_info")
            return builder.call(fn_tp, [string_array.num_items,
                string_array.num_total_chars, string_array.data,
                string_array.offsets, string_array.null_bitmap])

        # Numpy
        if isinstance(arr_type, types.Array):
            arr = context.make_array(arr_type)(context, builder, in_arr)
            assert arr_type.ndim == 1, "only 1D array shuffle supported"
            length = builder.extract_value(arr.shape, 0)
            typ_enum = _numba_to_c_type_map[arr_type.dtype]
            typ_arg = cgutils.alloca_once_value(
                builder, lir.Constant(lir.IntType(32), typ_enum))

            fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                [lir.IntType(64), lir.IntType(8).as_pointer(),
                lir.IntType(32)])
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="numpy_array_to_info")
            return builder.call(fn_tp, [length,
                builder.bitcast(arr.data, lir.IntType(8).as_pointer()),
                builder.load(typ_arg)])

    return array_info_type(arr_type), codegen
