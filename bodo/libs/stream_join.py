# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Support for streaming join (a.k.a. vectorized join).
This file is mostly wrappers for C++ implementations.
"""
from functools import cached_property
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
from bodo.utils.typing import (
    BodoError,
    MetaType,
    get_overload_const_bool,
    is_overload_bool,
    is_overload_none,
    raise_bodo_error,
    unwrap_typeref,
)
from bodo.utils.utils import numba_to_c_array_type, numba_to_c_type

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

    def __init__(
        self,
        build_key_inds,
        probe_key_inds,
        build_outer,
        probe_outer,
        build_table_type=types.unknown,
        probe_table_type=types.unknown,
    ):
        self.build_key_inds = build_key_inds
        self.probe_key_inds = probe_key_inds
        self.build_outer = build_outer
        self.probe_outer = probe_outer
        self.build_table_type = build_table_type
        self.probe_table_type = probe_table_type
        super().__init__(
            f"JoinStateType(build_keys={build_key_inds}, probe_keys={probe_key_inds}, build_outer={build_outer}, probe_outer={probe_outer}, build_table={build_table_type}, probe_table={probe_table_type})"
        )

    @property
    def key(self):
        return (
            self.build_key_inds,
            self.probe_key_inds,
            self.build_outer,
            self.probe_outer,
            self.build_table_type,
            self.probe_table_type,
        )

    @property
    def n_keys(self):
        return len(self.build_key_inds)

    # Methods used to compute the information needed for _init_join_state

    def _derive_cpp_indices(self, key_idxs, table_type):
        """Generate the indices used for the C++ table from the
        given Python table.

        Args:
            key_idxs (N Tuple(int)): The indices of the key columns
            table_type (TableType): The input table type.

        Returns:
            N Tuple(int): Tuple giving the order of the output indices
        """
        total_idxs = []
        for key_idx in key_idxs:
            total_idxs.append(key_idx)

        idx_set = set(key_idxs)
        for i in range(len(table_type.arr_types)):
            if i not in idx_set:
                total_idxs.append(i)
        return tuple(total_idxs)

    @cached_property
    def build_indices(self):
        if self.build_table_type == types.unknown:
            return ()
        else:
            return self._derive_cpp_indices(self.build_key_inds, self.build_table_type)

    @cached_property
    def probe_indices(self):
        if self.probe_table_type == types.unknown:
            return ()
        else:
            return self._derive_cpp_indices(self.probe_key_inds, self.probe_table_type)

    def _derive_c_types(self, key_idxs, table_type) -> np.ndarray:
        """Generate the CType Enum types for each array in the
        C++ build table via the indices.

        Args:
            key_idxs (N Tuple(int)): The indices of the key columns
            table_type (TableType): The input table type.

        Returns:
            List(int): List with the integer values of each CTypeEnum value.
        """
        return np.array(
            [numba_to_c_type(table_type.arr_types[i].dtype) for i in key_idxs],
            dtype=np.int8,
        )

    @cached_property
    def build_arr_ctypes(self) -> np.ndarray:
        """
        Fetch the CTypes used for each array in the build table.

        Note: We must use build_indices to account for reordering
        and/or duplicate keys.

        Returns:
            List(int): The ctypes for each array in the build table. Note
                that C++ wants the actual integer but these are the values derived from
                CTypeEnum.
        """
        indices = self.build_indices
        table = self.build_table_type
        if table == types.unknown:
            return np.array([], dtype=np.int8)
        else:
            return self._derive_c_types(indices, table)

    @cached_property
    def probe_arr_ctypes(self) -> np.ndarray:
        """
        Fetch the CTypes used for each array in the probe table.

        Note: We must use probe_indices to account for reordering
        and/or duplicate keys.

        Returns:
            List(int): The ctypes for each array in the probe table. Note
                that C++ wants the actual integer but these are the values derived from
                CTypeEnum.
        """
        indices = self.probe_indices
        table = self.probe_table_type
        if table == types.unknown:
            return np.array([], dtype=np.int8)
        else:
            return self._derive_c_types(indices, table)

    def _derive_c_array_types(self, key_idxs, table_type) -> np.ndarray:
        """Generate the CArrayTypeEnum Enum types for each array in the
        C++ build table via the indices.

        Args:
            key_idxs (N Tuple(int)): The indices of the key columns
            table_type (TableType): The input table type.

        Returns:
            List(int): List with the integer values of each CTypeEnum value.
        """
        return np.array(
            [numba_to_c_array_type(table_type.arr_types[i]) for i in key_idxs],
            dtype=np.int8,
        )

    @cached_property
    def build_arr_array_types(self) -> np.ndarray:
        """
        Fetch the CArrayTypeEnum used for each array in the build table.

        Note: We must use build_indices to account for reordering
        and/or duplicate keys.

        Returns:
            List(int): The CArrayTypeEnum for each array in the build table. Note
                that C++ wants the actual integer but these are the values derived from
                CArrayTypeEnum.
        """
        indices = self.build_indices
        table = self.build_table_type
        if table == types.unknown:
            return np.array([], dtype=np.int8)
        else:
            return self._derive_c_array_types(indices, table)

    @cached_property
    def probe_arr_array_types(self) -> np.ndarray:
        """
        Fetch the CArrayTypeEnum used for each array in the probe table.

        Note: We must use probe_indices to account for reordering
        and/or duplicate keys.

        Returns:
            List(int): The CArrayTypeEnum for each array in the probe table. Note
                that C++ wants the actual integer but these are the values derived from
                CArrayTypeEnum.
        """
        indices = self.probe_indices
        table = self.probe_table_type
        if table == types.unknown:
            return np.array([], dtype=np.int8)
        else:
            return self._derive_c_array_types(indices, table)

    @property
    def num_build_arrs(self) -> int:
        """
        Determine the number of build arrays.

        Note: We use build_indices in case the same column is used as a key in
        multiple comparisons.

        Return (int): The number of build arrays
        """
        return len(self.build_arr_ctypes)

    @property
    def num_probe_arrs(self) -> int:
        """
        Determine the number of probe arrays.

        Note: We use probe_indices in case the same column is used as a key in
        multiple comparisons.

        Return (int): The number of probe arrays
        """
        return len(self.probe_arr_ctypes)


register_model(JoinStateType)(models.OpaqueModel)


@intrinsic
def _init_join_state(
    typingctx,
    build_arr_dtypes,
    build_arr_array_types,
    n_build_arrs,
    probe_arr_dtypes,
    probe_arr_array_types,
    n_probe_arrs,
    output_state_type,
):
    """Initialize C++ JoinState pointer

    Args:
        build_arr_dtypes (int8*): pointer to array of ints representing array dtypes
                                   (as provided by numba_to_c_type)
        build_arr_array_types (int8*): pointer to array of ints representing array types
                                    (as provided by numba_to_c_array_type)
        n_build_arrs (int32): number of build columns
        probe_arr_dtypes (int8*): pointer to array of ints representing array dtypes
                                   (as provided by numba_to_c_type)
        probe_arr_array_types (int8*): pointer to array of ints representing array types
                                   (as provided by numba_to_c_array_type)
        n_probe_arrs (int32): number of probe columns
        output_state_type (TypeRef[JoinStateType]): The output type for the state that should be
                                                    generated.
    """
    output_type = unwrap_typeref(output_state_type)

    def codegen(context, builder, sig, args):
        (
            build_arr_dtypes,
            build_arr_array_types,
            n_build_arrs,
            probe_arr_dtypes,
            probe_arr_array_types,
            n_probe_arrs,
            _,
        ) = args
        n_keys = context.get_constant(types.int64, output_type.n_keys)
        build_table_outer = context.get_constant(types.bool_, output_type.build_outer)
        probe_table_outer = context.get_constant(types.bool_, output_type.probe_outer)
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

    sig = output_type(
        types.voidptr,
        types.voidptr,
        types.int32,
        types.voidptr,
        types.voidptr,
        types.int32,
        output_state_type,
    )
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def init_join_state(
    build_key_inds,
    probe_key_inds,
    build_outer,
    probe_outer,
    expected_state_type=None,
):
    expected_state_type = unwrap_typeref(expected_state_type)
    if is_overload_none(expected_state_type):
        build_keys = unwrap_typeref(build_key_inds).meta
        probe_keys = unwrap_typeref(probe_key_inds).meta
        if len(build_keys) != len(probe_keys):
            raise BodoError(
                "init_join_state(): Number of keys on build and probe sides must match"
            )
        if not (is_overload_bool(build_outer) and is_overload_bool(probe_outer)):
            raise_bodo_error(
                "init_join_state(): build_outer and probe_outer must be constant booleans"
            )

        output_type = JoinStateType(
            build_keys,
            probe_keys,
            get_overload_const_bool(build_outer),
            get_overload_const_bool(probe_outer),
        )
    else:
        output_type = expected_state_type

    build_arr_dtypes = output_type.build_arr_ctypes
    build_arr_array_types = output_type.build_arr_array_types
    n_build_arrs = output_type.num_build_arrs
    probe_arr_dtypes = output_type.probe_arr_ctypes
    probe_arr_array_types = output_type.probe_arr_array_types
    n_probe_arrs = output_type.num_probe_arrs

    def impl(
        build_key_inds,
        probe_key_inds,
        build_outer,
        probe_outer,
        expected_state_type=None,
    ):  # pragma: no cover
        return _init_join_state(
            build_arr_dtypes.ctypes,
            build_arr_array_types.ctypes,
            n_build_arrs,
            probe_arr_dtypes.ctypes,
            probe_arr_array_types.ctypes,
            n_probe_arrs,
            output_type,
        )

    return impl


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
    in_col_inds = MetaType(join_state.build_indices)
    n_table_cols = join_state.num_build_arrs

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
    in_col_inds = MetaType(join_state.probe_indices)
    n_table_cols = join_state.num_probe_arrs

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
