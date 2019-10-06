"""Tools for handling bodo arrays, e.g. passing to C/C++ code
"""
import numpy as np
import numba
from numba import types
from numba.extending import (typeof_impl, type_callable, models,
    register_model, NativeValue, make_attribute_wrapper, lower_builtin, box,
    unbox, lower_getattr, intrinsic, overload_method, overload,
    overload_attribute)
import bodo
from bodo.libs.str_arr_ext import string_array_type

from bodo.libs import array_tools_ext
from llvmlite import ir as lir
import llvmlite.binding as ll
ll.add_symbol('string_array_to_info', array_tools_ext.string_array_to_info)


class ArrayInfoType(types.Type):
    def __init__(self):
        super(ArrayInfoType, self).__init__(name='ArrayInfoType()')


array_info_type = ArrayInfoType()
register_model(ArrayInfoType)(models.OpaqueModel)


@intrinsic
def array_to_info(typingctx, arr_type):
    def codegen(context, builder, sig, args):
        in_str_arr, = args
        string_array = context.make_helper(
            builder, string_array_type, in_str_arr)
        fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
            [lir.IntType(64), lir.IntType(64), lir.IntType(8).as_pointer(),
            lir.IntType(32).as_pointer(), lir.IntType(8).as_pointer()])
        fn_tp = builder.module.get_or_insert_function(
            fnty, name="string_array_to_info")
        return builder.call(fn_tp, [string_array.num_items,
            string_array.num_total_chars, string_array.data,
            string_array.offsets, string_array.null_bitmap])

    return array_info_type(arr_type), codegen
