# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Support for streaming join (a.k.a. vectorized join).
This file is mostly wrappers for C++ implementations.
"""
from typing import TYPE_CHECKING

import llvmlite.binding as ll
import numba
import numpy as np
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic, models, register_model

import bodo
from bodo.ext import stream_join_cpp
from bodo.libs.array import (
    cpp_table_to_py_table,
    delete_table,
    py_data_to_cpp_table,
)
from bodo.utils.typing import BodoError, MetaType, unwrap_typeref

if TYPE_CHECKING:  # pragma: no cover
    pass


ll.add_symbol("join_state_init_py_entry", stream_join_cpp.join_state_init_py_entry)
ll.add_symbol(
    "join_build_consume_batch_py_entry",
    stream_join_cpp.join_build_consume_batch_py_entry,
)
ll.add_symbol(
    "join_probe_consume_batch_py_entry",
    stream_join_cpp.join_probe_consume_batch_py_entry,
)
ll.add_symbol("delete_join_state", stream_join_cpp.delete_join_state)


class JoinStateType(types.Type):
    """Type for C++ JoinState pointer"""

    def __init__(self, build_key_inds, probe_key_inds):
        self.build_key_inds = build_key_inds
        self.probe_key_inds = probe_key_inds
        super().__init__(
            f"JoinStateType(build_keys={build_key_inds}, probe_keys={probe_key_inds})"
        )


register_model(JoinStateType)(models.OpaqueModel)


@intrinsic
def init_join_state(
    typingctx,
    build_arr_dtypes,
    build_arr_array_types,
    n_build_arrs,
    probe_arr_dtypes,
    probe_arr_array_types,
    n_probe_arrs,
    build_key_inds,
    probe_key_inds,
    build_table_outer=False,
    probe_table_outer=False,
):
    """Initialize C++ JoinState pointer

    Args:
        build_arr_dtypes (int8*): pointer to array of ints representing array dtypes
                                   (as provided by numba_to_c_type)
        build_arr_array_types (int8*): pointer to array of ints representing array types
        n_build_arrs (int32): number of build columns
        probe_arr_dtypes (int8*): pointer to array of ints representing array dtypes
                                   (as provided by numba_to_c_type)
        probe_arr_array_types (int8*): pointer to array of ints representing array types
        n_probe_arrs (int32): number of probe columns
        build_key_inds (MetaType(NTuple(int64))): Column indices for the keys on the build side.
        probe_key_inds (MetaType(NTuple(int64))): Column indices for the keys on the probe side.
        build_table_outer (bool): whether to produce left outer join output
        probe_table_outer (bool): whether to produce right outer join output
    """
    build_keys = unwrap_typeref(build_key_inds).meta
    probe_keys = unwrap_typeref(probe_key_inds).meta
    if len(build_keys) != len(probe_keys):
        raise BodoError(
            "init_join_state(): Number of keys on build and probe sides must match"
        )

    def codegen(context, builder, sig, args):
        (
            build_arr_dtypes,
            build_arr_array_types,
            n_build_arrs,
            probe_arr_dtypes,
            probe_arr_array_types,
            n_probe_arrs,
            _,
            _,
            build_table_outer,
            probe_table_outer,
        ) = args
        n_keys = context.get_constant(types.int64, len(build_keys))
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(1),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="join_state_init_py_entry"
        )
        input_args = (
            build_arr_dtypes,
            build_arr_array_types,
            n_build_arrs,
            probe_arr_dtypes,
            probe_arr_array_types,
            n_probe_arrs,
            n_keys,
            build_table_outer,
            probe_table_outer,
        )
        ret = builder.call(fn_tp, input_args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    output_type = JoinStateType(build_keys, probe_keys)
    sig = output_type(
        types.voidptr,
        types.voidptr,
        types.int32,
        types.voidptr,
        types.voidptr,
        types.int32,
        build_key_inds,
        probe_key_inds,
        types.boolean,
        types.boolean,
    )
    return sig, codegen


@intrinsic
def _join_build_consume_batch(
    typingctx,
    join_state,
    cpp_table,
    is_last,
    parallel,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="join_build_consume_batch_py_entry"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    sig = types.void(join_state, cpp_table, is_last, parallel)
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def join_build_consume_batch(join_state, table, is_last, parallel=False):
    """Consume a build table batch in streaming join (insert into hash table)

    Args:
        join_state (JoinState): C++ JoinState pointer
        table (table_type): build table batch
        is_last (bool): is last batch
    """
    # Generate the new table type putting the keys in the front
    key_idxs = join_state.build_key_inds
    total_idxs = []
    for key_idx in key_idxs:
        total_idxs.append(key_idx)

    idx_set = set(key_idxs)
    for i in range(len(table.arr_types)):
        if i not in idx_set:
            total_idxs.append(i)

    in_col_inds = MetaType(tuple(total_idxs))
    n_table_cols = len(table.arr_types)

    def impl(join_state, table, is_last, parallel=False):  # pragma: no cover
        cpp_table = py_data_to_cpp_table(table, (), in_col_inds, n_table_cols)
        _join_build_consume_batch(join_state, cpp_table, is_last, parallel)

    return impl


@intrinsic
def _join_probe_consume_batch(
    typingctx,
    join_state,
    cpp_table,
    is_last,
    parallel,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="join_probe_consume_batch_py_entry"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = cpp_table(join_state, cpp_table, is_last, parallel)
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def join_probe_consume_batch(
    join_state, table, out_table_type, is_last, parallel=False
):
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
    # Generate the new table type putting the keys in the front
    key_idxs = join_state.probe_key_inds
    total_idxs = []
    for key_idx in key_idxs:
        total_idxs.append(key_idx)

    idx_set = set(key_idxs)
    for i in range(len(table.arr_types)):
        if i not in idx_set:
            total_idxs.append(i)

    in_col_inds = MetaType(tuple(total_idxs))
    n_table_cols = len(table.arr_types)

    n_out_arrs = len(unwrap_typeref(out_table_type).arr_types)

    def impl(
        join_state, table, out_table_type, is_last, parallel=False
    ):  # pragma: no cover
        cpp_table = py_data_to_cpp_table(table, (), in_col_inds, n_table_cols)
        out_cpp_table = _join_probe_consume_batch(
            join_state, cpp_table, is_last, parallel
        )
        out_table = cpp_table_to_py_table(
            out_cpp_table, np.arange(n_out_arrs), out_table_type
        )
        delete_table(out_cpp_table)
        return out_table

    return impl


@intrinsic
def delete_join_state(
    typingctx,
    join_state,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="delete_join_state"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    sig = types.void(join_state)
    return sig, codegen
