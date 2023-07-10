# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""State and API information for using dictionary encoded arrays
in a streaming fashion with a goal of minimizing the amount of computation.
These implementations are focused on SQL Projection and Filter operations
with a goal of caching computation if a dictionary has already been encountered.

For more information check the confluence design doc:
https://bodo.atlassian.net/wiki/spaces/B/pages/1402175534/Dictionary+Encoding+Parfors
"""
import llvmlite.binding as ll
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic, models, register_jitable, register_model

import bodo
from bodo.ext import stream_dict_encoding_cpp
from bodo.libs.array import (
    array_info_type,
    array_to_info,
    delete_info,
    info_to_array,
)

ll.add_symbol(
    "dict_encoding_state_init_py_entry",
    stream_dict_encoding_cpp.dict_encoding_state_init_py_entry,
)
ll.add_symbol(
    "state_contains_dict_array",
    stream_dict_encoding_cpp.state_contains_dict_array,
)
ll.add_symbol(
    "get_array_py_entry",
    stream_dict_encoding_cpp.get_array_py_entry,
)
ll.add_symbol(
    "set_array_py_entry",
    stream_dict_encoding_cpp.set_array_py_entry,
)
ll.add_symbol(
    "delete_dict_encoding_state", stream_dict_encoding_cpp.delete_dict_encoding_state
)


class DictionaryEncodingStateType(types.Type):
    def __init__(self):
        super().__init__(f"DictionaryEncodingStateType()")


dictionary_encoding_state_type = DictionaryEncodingStateType()
register_model(DictionaryEncodingStateType)(models.OpaqueModel)


@intrinsic
def init_dict_encoding_state(typingctx):
    """Initialize the C++ DictionaryEncodingState pointer"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.IntType(8).as_pointer(), [])
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="dict_encoding_state_init_py_entry"
        )
        ret = builder.call(fn_tp, ())
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = dictionary_encoding_state_type()
    return sig, codegen


@intrinsic
def state_contains_dict_array(typingctx, dict_encoding_state, func_id, dict_id):
    """Return if the given dictionary encoding state has cached
    the result of the given function with the given dictionary.

    Args:
        dict_encoding_state (DictionaryEncodingStateType): The state to check.
        func_id (types.int64): Unique id for the function.
        dict_id (types.int64): Unique id for the input array to check.

    Returns:
        types.bool_: Does the state definitely contain the array.
        This can have false negatives (arrays are the same but have
        different ids), but no false positives.
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(1),
            [lir.IntType(8).as_pointer(), lir.IntType(64), lir.IntType(64)],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="state_contains_dict_array"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.bool_(dict_encoding_state, types.int64, types.int64)
    return sig, codegen


@register_jitable
def get_array(dict_encoding_state, func_id, cache_dict_id):  # pragma: no cover
    arr_info, new_dict_id = _get_array(dict_encoding_state, func_id, cache_dict_id)
    arr = info_to_array(arr_info, bodo.string_array_type)
    delete_info(arr_info)
    return (arr, new_dict_id)


@intrinsic
def _get_array(typingctx, dict_encoding_state, func_id, cache_dict_id):
    def codegen(context, builder, sig, args):
        dict_encoding_state, func_id, dict_id = args
        # Generate pointer for loading data from C++
        new_dict_id_ptr = cgutils.alloca_once_value(
            builder, context.get_constant(types.int64, -1)
        )
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="get_array_py_entry"
        )
        call_args = [dict_encoding_state, func_id, dict_id, new_dict_id_ptr]
        arr_info = builder.call(fn_tp, call_args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        new_dict_id = builder.load(new_dict_id_ptr)
        return context.make_tuple(builder, sig.return_type, [arr_info, new_dict_id])

    sig = types.Tuple([array_info_type, types.int64])(
        dict_encoding_state, types.int64, types.int64
    )
    return sig, codegen


@register_jitable
def set_array(
    dict_encoding_state, func_id, cache_dict_id, arr, new_dict_id
):  # pragma: no cover
    arr_info = array_to_info(arr)
    _set_array(dict_encoding_state, func_id, cache_dict_id, arr_info, new_dict_id)


@intrinsic
def _set_array(
    typingctx, dict_encoding_state, func_id, cache_dict_id, arr_info, new_dict_id
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="set_array_py_entry"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    sig = types.void(
        dict_encoding_state, types.int64, types.int64, arr_info, types.int64
    )
    return sig, codegen


@intrinsic
def delete_dict_encoding_state(typingctx, dict_encoding_state):
    """Initialize the C++ DictionaryEncodingState pointer"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="delete_dict_encoding_state"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    sig = types.void(dict_encoding_state)
    return sig, codegen
