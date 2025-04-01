"""State and API information for using dictionary encoded arrays
in a streaming fashion with a goal of minimizing the amount of computation.
These implementations are focused on SQL Projection and Filter operations
with a goal of caching computation if a dictionary has already been encountered.

For more information check the confluence design doc:
https://bodo.atlassian.net/wiki/spaces/B/pages/1402175534/Dictionary+Encoding+Parfors
"""

import llvmlite.binding as ll
import numba
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
    "state_contains_multi_input_dict_array",
    stream_dict_encoding_cpp.state_contains_multi_input_dict_array,
)
ll.add_symbol(
    "get_array_multi_input_py_entry",
    stream_dict_encoding_cpp.get_array_multi_input_py_entry,
)
ll.add_symbol(
    "set_array_multi_input_py_entry",
    stream_dict_encoding_cpp.set_array_multi_input_py_entry,
)
ll.add_symbol(
    "get_state_num_set_calls",
    stream_dict_encoding_cpp.get_state_num_set_calls,
)
ll.add_symbol(
    "delete_dict_encoding_state", stream_dict_encoding_cpp.delete_dict_encoding_state
)


class DictionaryEncodingStateType(types.Type):
    def __init__(self):
        super().__init__("DictionaryEncodingStateType()")


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
            lir.IntType(64),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="state_contains_dict_array"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.int64(dict_encoding_state, types.int64, types.int64)
    return sig, codegen


@register_jitable
def get_array(
    dict_encoding_state, func_id, cache_dict_id, array_ref_type
):  # pragma: no cover
    arr_info, new_dict_id, cached_dict_length = _get_array(
        dict_encoding_state, func_id, cache_dict_id
    )
    arr = info_to_array(arr_info, array_ref_type)
    delete_info(arr_info)
    return (arr, new_dict_id, cached_dict_length)


@intrinsic
def _get_array(typingctx, dict_encoding_state, func_id, cache_dict_id):
    def codegen(context, builder, sig, args):
        dict_encoding_state, func_id, dict_id = args
        # Generate pointer for loading data from C++
        new_dict_id_ptr = cgutils.alloca_once_value(
            builder, context.get_constant(types.int64, -1)
        )
        cached_dict_length_ptr = cgutils.alloca_once_value(
            builder, context.get_constant(types.int64, -1)
        )
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64).as_pointer(),
                lir.IntType(64).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="get_array_py_entry"
        )
        call_args = [
            dict_encoding_state,
            func_id,
            dict_id,
            new_dict_id_ptr,
            cached_dict_length_ptr,
        ]
        arr_info = builder.call(fn_tp, call_args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        new_dict_id = builder.load(new_dict_id_ptr)
        cached_dict_length = builder.load(cached_dict_length_ptr)
        return context.make_tuple(
            builder, sig.return_type, [arr_info, new_dict_id, cached_dict_length]
        )

    sig = types.Tuple([array_info_type, types.int64, types.int64])(
        dict_encoding_state, types.int64, types.int64
    )
    return sig, codegen


@register_jitable
def set_array(
    dict_encoding_state, func_id, cache_dict_id, cache_dict_length, arr, new_dict_id
):  # pragma: no cover
    arr_info = array_to_info(arr)
    _set_array(
        dict_encoding_state,
        func_id,
        cache_dict_id,
        cache_dict_length,
        arr_info,
        new_dict_id,
    )


@intrinsic
def _set_array(
    typingctx,
    dict_encoding_state,
    func_id,
    cache_dict_id,
    cache_dict_length,
    arr_info,
    new_dict_id,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
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
        dict_encoding_state,
        types.int64,
        types.int64,
        types.int64,
        arr_info,
        types.int64,
    )
    return sig, codegen


def _get_list_payload(context, builder, list_type, list_value):
    list_struct = cgutils.create_struct_proxy(list_type)(
        context, builder, value=list_value
    )
    return numba.cpython.listobj.get_list_payload(
        context, builder, list_type, list_struct
    )


@intrinsic
def state_contains_multi_input_dict_array(
    typingctx, dict_encoding_state, func_id, dict_ids, dict_lens
):
    """Return if the given dictionary encoding state has cached
    the result of the given function with the given multi-dictionary
    input function.

    Args:
        dict_encoding_state (DictionaryEncodingStateType): The state to check.
        func_id (types.int64): Unique id for the function.
        dict_id (types.ListType[types.int64]): Unique ids for the input arrays to check.

    Returns:
        types.bool_: Does the state definitely contain the array.
        This can have false negatives (arrays are the same but have
        different ids), but no false positives.
    """
    assert isinstance(dict_ids, types.List) and dict_ids.dtype == types.int64, (
        "dict_ids must a be list of int64"
    )
    assert isinstance(dict_lens, types.List) and dict_lens.dtype == types.int64, (
        "dict_lens must a be list of int64"
    )

    def codegen(context, builder, sig, args):
        dict_encoding_state, func_id, dict_ids, dict_lens = args
        fnty = lir.FunctionType(
            lir.IntType(1),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64).as_pointer(),
                lir.IntType(64).as_pointer(),
                lir.IntType(64),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="state_contains_multi_input_dict_array"
        )

        ids_payload_struct = _get_list_payload(context, builder, sig.args[2], dict_ids)
        lens_payload_struct = _get_list_payload(
            context, builder, sig.args[3], dict_lens
        )
        call_args = [
            dict_encoding_state,
            func_id,
            ids_payload_struct._get_ptr_by_name("data"),
            lens_payload_struct._get_ptr_by_name("data"),
            ids_payload_struct.size,
        ]
        ret = builder.call(fn_tp, call_args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.bool_(dict_encoding_state, types.int64, dict_ids, dict_lens)
    return sig, codegen


@register_jitable
def get_array_multi_input(
    dict_encoding_state, func_id, cache_dict_ids, cache_dict_lens, array_ref_type
):  # pragma: no cover
    arr_info, new_dict_id = _get_array_multi_input(
        dict_encoding_state, func_id, cache_dict_ids, cache_dict_lens
    )
    arr = info_to_array(arr_info, array_ref_type)
    delete_info(arr_info)
    return (arr, new_dict_id)


@intrinsic
def _get_array_multi_input(
    typingctx, dict_encoding_state, func_id, cache_dict_ids, cache_dict_lens
):
    assert (
        isinstance(cache_dict_ids, types.List) and cache_dict_ids.dtype == types.int64
    ), "dict_ids must a be list of int64"
    assert (
        isinstance(cache_dict_lens, types.List) and cache_dict_lens.dtype == types.int64
    ), "dict_len must a be list of int64"

    def codegen(context, builder, sig, args):
        dict_encoding_state, func_id, cache_dict_ids, cache_dict_lens = args
        # Generate pointer for loading data from C++
        new_dict_id_ptr = cgutils.alloca_once_value(
            builder, context.get_constant(types.int64, -1)
        )
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64).as_pointer(),
                lir.IntType(64).as_pointer(),
                lir.IntType(64),
                lir.IntType(64).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="get_array_multi_input_py_entry"
        )
        ids_payload_struct = _get_list_payload(
            context, builder, sig.args[2], cache_dict_ids
        )
        lens_payload_struct = _get_list_payload(
            context, builder, sig.args[3], cache_dict_lens
        )
        call_args = [
            dict_encoding_state,
            func_id,
            ids_payload_struct._get_ptr_by_name("data"),
            lens_payload_struct._get_ptr_by_name("data"),
            ids_payload_struct.size,
            new_dict_id_ptr,
        ]
        arr_info = builder.call(fn_tp, call_args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        new_dict_id = builder.load(new_dict_id_ptr)
        return context.make_tuple(builder, sig.return_type, [arr_info, new_dict_id])

    sig = types.Tuple([array_info_type, types.int64])(
        dict_encoding_state, types.int64, cache_dict_ids, cache_dict_lens
    )
    return sig, codegen


@register_jitable
def set_array_multi_input(
    dict_encoding_state, func_id, cache_dict_ids, cache_dict_lens, arr, new_dict_id
):  # pragma: no cover
    arr_info = array_to_info(arr)
    _set_array_multi_input(
        dict_encoding_state,
        func_id,
        cache_dict_ids,
        cache_dict_lens,
        arr_info,
        new_dict_id,
    )


@intrinsic
def _set_array_multi_input(
    typingctx,
    dict_encoding_state,
    func_id,
    cache_dict_ids,
    cache_dict_lens,
    arr_info,
    new_dict_id,
):
    assert (
        isinstance(cache_dict_ids, types.List) and cache_dict_ids.dtype == types.int64
    ), "dict_ids must a be list of int64"
    assert (
        isinstance(cache_dict_lens, types.List) and cache_dict_lens.dtype == types.int64
    ), "dict_len must a be list of int64"

    def codegen(context, builder, sig, args):
        (
            dict_encoding_state,
            func_id,
            cache_dict_ids,
            cache_dict_lens,
            arr_info,
            new_dict_id,
        ) = args
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64).as_pointer(),
                lir.IntType(64).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
            ],
        )
        ids_payload_struct = _get_list_payload(
            context, builder, sig.args[2], cache_dict_ids
        )
        lens_payload_struct = _get_list_payload(
            context, builder, sig.args[3], cache_dict_lens
        )
        call_args = [
            dict_encoding_state,
            func_id,
            ids_payload_struct._get_ptr_by_name("data"),
            lens_payload_struct._get_ptr_by_name("data"),
            ids_payload_struct.size,
            arr_info,
            new_dict_id,
        ]
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="set_array_multi_input_py_entry"
        )
        builder.call(fn_tp, call_args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    sig = types.void(
        dict_encoding_state,
        types.int64,
        cache_dict_ids,
        cache_dict_lens,
        arr_info,
        types.int64,
    )
    return sig, codegen


@intrinsic
def get_state_num_set_calls(typingctx, dict_encoding_state):
    """Get the number of times set was called on the dictionary encoding state."""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(8).as_pointer()])
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="get_state_num_set_calls"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.int64(dict_encoding_state)
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
