# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Support for streaming join (a.k.a. vectorized join).
This file is mostly wrappers for C++ implementations.
"""
from typing import TYPE_CHECKING

import llvmlite.binding as ll
import numba
import numpy as np
from numba.core import types
from numba.extending import models, register_model

import bodo
from bodo.ext import stream_join_cpp
from bodo.libs.array import (
    cpp_table_to_py_table,
    delete_table,
    py_table_to_cpp_table,
)
from bodo.libs.array import table_type as cpp_table_type
from bodo.utils.typing import unwrap_typeref

if TYPE_CHECKING:  # pragma: no cover
    pass


ll.add_symbol("join_state_init_py_entry", stream_join_cpp.join_state_init_py_entry)
ll.add_symbol(
    "join_build_get_batch_py_entry", stream_join_cpp.join_build_get_batch_py_entry
)
ll.add_symbol(
    "join_probe_get_batch_py_entry", stream_join_cpp.join_probe_get_batch_py_entry
)
ll.add_symbol("delete_join_state", stream_join_cpp.delete_join_state)


class JoinStateType(types.Type):
    """Opaque type for C++ JoinState pointer"""

    def __init__(self):
        super().__init__(f"JoinStateType()")


register_model(JoinStateType)(models.OpaqueModel)
join_state_type = JoinStateType()


_init_join_state = types.ExternalFunction(
    "join_state_init_py_entry",
    join_state_type(types.voidptr, types.int32, types.int64),
)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def init_join_state(build_arr_types, n_arrs, n_keys):
    """Initialize C++ JoinState pointer

    Args:
        build_arr_types_t (int8*): pointer to array of ints representing array types
                                   (as provided by numba_to_c_type)
        n_arrs_t (int32): number of build columns
        n_keys_t (int64): number of keys (assuming key columns are first in build table)
    """

    def impl(build_arr_types, n_arrs, n_keys):  # pragma: no cover
        join_state = _init_join_state(build_arr_types, n_arrs, n_keys)
        bodo.utils.utils.check_and_propagate_cpp_exception()
        return join_state

    return impl


_join_build_get_batch = types.ExternalFunction(
    "join_build_get_batch_py_entry",
    types.void(join_state_type, cpp_table_type, types.bool_),
)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def join_build_get_batch(join_state, table, is_last):
    """Consume a build table batch in streaming join (insert into hash table)

    Args:
        join_state (JoinState): C++ JoinState pointer
        table (table_type): build table batch
        is_last (bool): is last batch
    """
    table_type = table

    def impl(join_state, table, is_last):  # pragma: no cover
        cpp_table = py_table_to_cpp_table(table, table_type)
        _join_build_get_batch(join_state, cpp_table, is_last)
        bodo.utils.utils.check_and_propagate_cpp_exception()

    return impl


_join_probe_get_batch = types.ExternalFunction(
    "join_probe_get_batch_py_entry",
    cpp_table_type(join_state_type, cpp_table_type, types.bool_),
)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def join_probe_get_batch(join_state, table, out_table_type, is_last):
    """Consume a probe table batch in streaming join (probe hash table and produce
    output rows)

    Args:
        join_state (JoinState): C++ JoinState pointer
        table (table_type): probe table batch
        out_table_type (table_type|TypeRef): full type of output table batches
        is_last (bool): is last batch

    Returns:
        table_type: output table batch
    """
    in_table_type = table
    n_out_arrs = len(unwrap_typeref(out_table_type).arr_types)

    def impl(join_state, table, out_table_type, is_last):  # pragma: no cover
        cpp_table = py_table_to_cpp_table(table, in_table_type)
        out_cpp_table = _join_probe_get_batch(join_state, cpp_table, is_last)
        bodo.utils.utils.check_and_propagate_cpp_exception()
        out_table = cpp_table_to_py_table(
            out_cpp_table, np.arange(n_out_arrs), out_table_type
        )
        delete_table(out_cpp_table)
        return out_table

    return impl


delete_join_state = types.ExternalFunction(
    "delete_join_state",
    types.void(join_state_type),
)
