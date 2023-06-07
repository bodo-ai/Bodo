# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Support for streaming join (a.k.a. vectorized join).
This file is mostly wrappers for C++ implementations.
"""
from functools import cached_property
from typing import TYPE_CHECKING, List

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
    get_common_bodosql_integer_arr_type,
    get_overload_const_bool,
    get_overload_const_str,
    is_bodosql_integer_arr_type,
    is_nullable,
    is_overload_bool,
    is_overload_none,
    raise_bodo_error,
    to_nullable_type,
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
        build_column_names,
        probe_column_names,
        build_outer,
        probe_outer,
        build_table_type=types.unknown,
        probe_table_type=types.unknown,
    ):
        self.build_key_inds = build_key_inds
        self.probe_key_inds = probe_key_inds
        self.build_column_names = build_column_names
        self.probe_column_names = probe_column_names
        self.build_outer = build_outer
        self.probe_outer = probe_outer
        self.build_table_type = build_table_type
        self.probe_table_type = probe_table_type
        super().__init__(
            (
                f"JoinStateType("
                f"build_keys={build_key_inds}, "
                f"probe_keys={probe_key_inds}, "
                f"build_column_names={build_column_names}, "
                f"probe_column_names={probe_column_names}, "
                f"build_outer={build_outer}, "
                f"probe_outer={probe_outer}, "
                f"build_table={build_table_type}, "
                f"probe_table={probe_table_type})"
            )
        )

    @property
    def key(self):
        return (
            self.build_key_inds,
            self.probe_key_inds,
            self.build_column_names,
            self.probe_column_names,
            self.build_outer,
            self.probe_outer,
            self.build_table_type,
            self.probe_table_type,
        )

    @property
    def n_keys(self):
        return len(self.build_key_inds)

    # Methods used to compute the information needed for _init_join_state

    def _derive_cpp_indices(self, key_indices, num_cols):
        """Generate the indices used for the C++ table from the
        given Python table.

        Args:
            key_indices (N Tuple(int)): The indices of the key columns
            num_cols (int): The number of total columns in the array.

        Returns:
            N Tuple(int): Tuple giving the order of the output indices
        """
        total_idxs = []
        for key_idx in key_indices:
            total_idxs.append(key_idx)

        idx_set = set(key_indices)
        for i in range(num_cols):
            if i not in idx_set:
                total_idxs.append(i)
        return tuple(total_idxs)

    @cached_property
    def build_indices(self):
        if self.build_table_type == types.unknown:
            return ()
        else:
            return self._derive_cpp_indices(
                self.build_key_inds, len(self.build_table_type.arr_types)
            )

    @cached_property
    def probe_indices(self):
        if self.probe_table_type == types.unknown:
            return ()
        else:
            return self._derive_cpp_indices(
                self.probe_key_inds, len(self.probe_table_type.arr_types)
            )

    def _derive_input_type(
        self, key_types, key_indices, table_type
    ) -> List[types.ArrayCompatible]:
        """Generate the input table type based on the given key types, key
        indices, and table type.

        Args:
            key_types (List[types.ArrayCompatible]): The list of key types in order.
            key_indices (N Tuple(int)): The indices of the key columns
            table_type (TableType): The input table type.

        Returns:
            List[types.ArrayCompatible]: The list of array types for the input table (in order).
        """
        types = key_types.copy()
        idx_set = set(key_indices)
        for i in range(len(table_type.arr_types)):
            # Append the data columns.
            if i not in idx_set:
                types.append(table_type.arr_types[i])
        return types

    @cached_property
    def key_types(self) -> List[types.ArrayCompatible]:
        """Generate the list of array types that should be used for the
        keys to hash join. For build/probe the keys must match exactly.

        Returns:
            List[types.ArrayCompatible]: The list of array types used
            by the keys.
        """
        build_table_type = self.build_table_type
        probe_table_type = self.probe_table_type
        if build_table_type == types.unknown or probe_table_type == types.unknown:
            # This path may be reached if the typing transformations haven't fully finished.
            return []
        build_key_inds = self.build_key_inds
        probe_key_inds = self.probe_key_inds
        arr_types = []
        num_keys = len(build_key_inds)
        for i in range(num_keys):
            build_arr_type = self.build_table_type.arr_types[build_key_inds[i]]
            probe_arr_type = self.probe_table_type.arr_types[probe_key_inds[i]]
            probe_nullable = is_nullable(probe_arr_type)
            build_nullable = is_nullable(build_arr_type)
            # Convert arrays if they aren't the same nullability.
            if probe_nullable != build_nullable:
                if probe_nullable:
                    build_arr_type = to_nullable_type(build_arr_type)
                if build_nullable:
                    probe_arr_type = to_nullable_type(probe_arr_type)
            if build_arr_type == probe_arr_type:
                key_type = build_arr_type
            elif is_bodosql_integer_arr_type(
                build_arr_type
            ) and is_bodosql_integer_arr_type(probe_arr_type):
                # TODO: Future optimization. If the types don't match exactly and
                # we don't have an outer join on the larger bitwidth side, we don't
                # have to upcast and can instead filter + downcast. In particular we
                # know that any type that doesn't fit in the smaller array type
                # cannot have a match.
                #
                # Note: If we have an outer join we can't do this because we need to
                # preserve the elements without matches.
                #
                key_type = get_common_bodosql_integer_arr_type(
                    build_arr_type, probe_arr_type
                )
            else:
                # TODO [BSE-439]: Support dict encoding + regular string
                raise BodoError(
                    "StreamingHashJoin: Build and probe keys must have the same types"
                )
            arr_types.append(key_type)
        return arr_types

    def _key_casted_table_type(self, key_types, key_indices, table_type):
        """
        Generate the table type produced by only casting
        the keys to the shared key type and keeping all data columns the
        same.

        Args:
            key_types (List[types.ArrayCompatible]): The list of array types used
                by the keys.
            key_indices (N Tuple(int)): The indices of the key columns
            table_type (TableType): The table type without casting.

        Returns:
            TableType: The new table type.
        """

        # Assume most cases don't cast.
        should_cast = False
        for i, idx in enumerate(key_indices):
            if key_types[i] != table_type.arr_types[idx]:
                should_cast = True
                break
        if not should_cast:
            # No need to generate a new type
            return table_type
        indices_map = {idx: i for i, idx in enumerate(key_indices)}
        arr_types = []
        for i in range(len(table_type.arr_types)):
            if i in indices_map:
                arr_types.append(key_types[indices_map[i]])
            else:
                arr_types.append(table_type.arr_types[i])
        return bodo.TableType(tuple(arr_types))

    @property
    def key_casted_build_table_type(self):
        """Generate the table type produced by only casting
        the build keys to the shared key type and keeping all
        data columns the same.

        Returns:
            TableType: The new table type.
        """
        if (
            self.build_table_type == types.unknown
            or self.probe_table_type == types.unknown
        ):
            return self.build_table_type
        return self._key_casted_table_type(
            self.key_types, self.build_key_inds, self.build_table_type
        )

    @property
    def key_casted_probe_table_type(self):
        """Generate the table type produced by only casting
        the probe keys to the shared key type and keeping all
        data columns the same.

        Returns:
            TableType: The new table type.
        """
        if (
            self.build_table_type == types.unknown
            or self.probe_table_type == types.unknown
        ):
            return self.probe_table_type
        return self._key_casted_table_type(
            self.key_types, self.probe_key_inds, self.probe_table_type
        )

    @cached_property
    def build_reordered_arr_types(self) -> List[types.ArrayCompatible]:
        """
        Get the list of array types for the actual input to the C++ build table.
        This is different from the build_table_type because the input to the C++
        will reorder keys to the front and may cast keys to matching types.

        Returns:
            List[types.ArrayCompatible]: The list of array types for the build table.
        """
        if self.build_table_type == types.unknown:
            return []
        key_types = self.key_types
        key_indices = self.build_key_inds
        table = self.build_table_type
        return self._derive_input_type(key_types, key_indices, table)

    @cached_property
    def probe_reordered_arr_types(self) -> List[types.ArrayCompatible]:
        """
        Get the list of array types for the actual input to the C++ probe table.
        This is different from the probe_table_type because the input to the C++
        will reorder keys to the front and may cast keys to matching types.

        Returns:
            List[types.ArrayCompatible]: The list of array types for the probe table.
        """
        if self.probe_table_type == types.unknown:
            return []
        key_types = self.key_types
        key_indices = self.probe_key_inds
        table = self.probe_table_type
        return self._derive_input_type(key_types, key_indices, table)

    def _derive_c_types(self, arr_types: List[types.ArrayCompatible]) -> np.ndarray:
        """Generate the CType Enum types for each array in the
        C++ build table via the indices.

        Args:
            arr_types (List[types.ArrayCompatible]): The array types to use.

        Returns:
            List(int): List with the integer values of each CTypeEnum value.
        """
        return np.array(
            [numba_to_c_type(arr_type.dtype) for arr_type in arr_types],
            dtype=np.int8,
        )

    @property
    def build_arr_ctypes(self) -> np.ndarray:
        """
        Fetch the CTypes used for each array in the build table.

        Note: We must use build_reordered_arr_types to account for reordering
        and/or cast keys.

        Returns:
            List(int): The ctypes for each array in the build table. Note
                that C++ wants the actual integer but these are the values derived from
                CTypeEnum.
        """
        return self._derive_c_types(self.build_reordered_arr_types)

    @property
    def probe_arr_ctypes(self) -> np.ndarray:
        """
        Fetch the CTypes used for each array in the probe table.

        Note: We must use probe_reordered_arr_types to account for reordering
        and/or cast keys.

        Returns:
            List(int): The ctypes for each array in the probe table. Note
                that C++ wants the actual integer but these are the values derived from
                CTypeEnum.
        """
        return self._derive_c_types(self.probe_reordered_arr_types)

    def _derive_c_array_types(
        self, arr_types: List[types.ArrayCompatible]
    ) -> np.ndarray:
        """Generate the CArrayTypeEnum Enum types for each array in the
        C++ build table via the indices.

        Args:
            arr_types (List[types.ArrayCompatible]): The array types to use.

        Returns:
            List(int): List with the integer values of each CTypeEnum value.
        """
        return np.array(
            [numba_to_c_array_type(arr_type) for arr_type in arr_types],
            dtype=np.int8,
        )

    @property
    def build_arr_array_types(self) -> np.ndarray:
        """
        Fetch the CArrayTypeEnum used for each array in the build table.

        Note: We must use build_reordered_arr_types to account for reordering
        and/or cast keys.


        Returns:
            List(int): The CArrayTypeEnum for each array in the build table. Note
                that C++ wants the actual integer but these are the values derived from
                CArrayTypeEnum.
        """
        return self._derive_c_array_types(self.build_reordered_arr_types)

    @property
    def probe_arr_array_types(self) -> np.ndarray:
        """
        Fetch the CArrayTypeEnum used for each array in the probe table.

        Note: We must use probe_reordered_arr_types to account for reordering
        and/or cast keys.


        Returns:
            List(int): The CArrayTypeEnum for each array in the probe table. Note
                that C++ wants the actual integer but these are the values derived from
                CArrayTypeEnum.
        """
        return self._derive_c_array_types(self.probe_reordered_arr_types)

    @property
    def num_build_arrs(self) -> int:
        """
        Determine the number of build arrays.

        Note: We use build_indices in case the same column is used as a key in
        multiple comparisons.

        Return (int): The number of build arrays
        """
        return len(self.build_reordered_arr_types)

    @property
    def num_probe_arrs(self) -> int:
        """
        Determine the number of probe arrays.

        Note: We use probe_indices in case the same column is used as a key in
        multiple comparisons.

        Return (int): The number of probe arrays
        """
        return len(self.probe_reordered_arr_types)

    @property
    def output_type(self):
        """Return the output type from generating this join.

        Note: We must use build_arr_types and probe_arr_types to
        account for key reordering + casting.

        Returns:
            bodo.TableType: The type of the output table.
        """
        arr_types = []
        # Note: We can maintain the input key types in the output.
        # Nothing should assume the original type if we cast an integer.
        key_types = self.key_types
        # Add build side
        for arr_type in key_types + self.build_reordered_arr_types[self.n_keys :]:
            if self.probe_outer:
                arr_type = to_nullable_type(arr_type)
            arr_types.append(arr_type)
        # Add probe side
        for arr_type in key_types + self.probe_reordered_arr_types[self.n_keys :]:
            if self.build_outer:
                arr_type = to_nullable_type(arr_type)
            arr_types.append(arr_type)

        out_table_type = bodo.TableType(tuple(arr_types))
        return out_table_type


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
    cfunc_cond_t,
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
            cfunc_cond,
        ) = args
        n_keys = context.get_constant(types.uint64, output_type.n_keys)
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
                lir.IntType(8).as_pointer(),
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
            cfunc_cond,
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
        types.voidptr,
    )
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def init_join_state(
    build_key_inds,
    probe_key_inds,
    build_colnames,
    probe_colnames,
    build_outer,
    probe_outer,
    expected_state_type=None,
    # The non-equality portion of the join condition. If None then
    # the join is a pure hash join. Otherwise this is a string similar
    # to the query string accepted by merge.
    non_equi_condition=None,
):
    expected_state_type = unwrap_typeref(expected_state_type)
    if is_overload_none(expected_state_type):
        build_keys = unwrap_typeref(build_key_inds).meta
        probe_keys = unwrap_typeref(probe_key_inds).meta
        build_column_names = unwrap_typeref(build_colnames).meta
        probe_column_names = unwrap_typeref(probe_colnames).meta
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
            build_column_names,
            probe_column_names,
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

    # handle non-equi conditions (reuse existing join code as much as possible)
    # Note we must account for how keys will be cast here.
    build_table_type = output_type.key_casted_build_table_type
    probe_table_type = output_type.key_casted_probe_table_type

    if (
        not is_overload_none(non_equi_condition)
        and build_table_type != types.unknown
        and probe_table_type != types.unknown
    ):
        from bodo.ir.join import (
            add_join_gen_cond_cfunc_sym,
            gen_general_cond_cfunc,
            get_join_cond_addr,
        )

        gen_expr_const = get_overload_const_str(non_equi_condition)

        # Parse the query
        _, _, parsed_gen_expr = bodo.hiframes.dataframe_impl._parse_merge_cond(
            gen_expr_const,
            output_type.build_column_names,
            build_table_type.arr_types,
            output_type.probe_column_names,
            probe_table_type.arr_types,
        )

        left_logical_to_physical = output_type.build_indices
        right_logical_to_physical = output_type.probe_indices
        left_var_map = {c: i for i, c in enumerate(output_type.build_column_names)}
        right_var_map = {c: i for i, c in enumerate(output_type.probe_column_names)}

        # Generate a general join condition cfunc
        general_cond_cfunc, _, _ = gen_general_cond_cfunc(
            None,
            left_logical_to_physical,
            right_logical_to_physical,
            str(parsed_gen_expr),
            left_var_map,
            None,
            set(),
            build_table_type,
            right_var_map,
            None,
            set(),
            probe_table_type,
            compute_in_batch=not output_type.build_key_inds,
        )
        cfunc_native_name = general_cond_cfunc.native_name

        def impl_nonequi(
            build_key_inds,
            probe_key_inds,
            build_colnames,
            probe_colnames,
            build_outer,
            probe_outer,
            expected_state_type=None,
            non_equi_condition=None,
        ):  # pragma: no cover
            cfunc_cond = add_join_gen_cond_cfunc_sym(
                general_cond_cfunc, cfunc_native_name
            )
            cfunc_cond = get_join_cond_addr(cfunc_native_name)
            return _init_join_state(
                build_arr_dtypes.ctypes,
                build_arr_array_types.ctypes,
                n_build_arrs,
                probe_arr_dtypes.ctypes,
                probe_arr_array_types.ctypes,
                n_probe_arrs,
                output_type,
                cfunc_cond,
            )

        return impl_nonequi

    def impl(
        build_key_inds,
        probe_key_inds,
        build_colnames,
        probe_colnames,
        build_outer,
        probe_outer,
        expected_state_type=None,
        non_equi_condition=None,
    ):  # pragma: no cover
        return _init_join_state(
            build_arr_dtypes.ctypes,
            build_arr_array_types.ctypes,
            n_build_arrs,
            probe_arr_dtypes.ctypes,
            probe_arr_array_types.ctypes,
            n_probe_arrs,
            output_type,
            0,
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
    cast_table_type = join_state.key_casted_build_table_type
    if cast_table_type == types.unknown:
        cast_table_type = table

    def impl(join_state, table, is_last, parallel=False):  # pragma: no cover
        cast_table = bodo.utils.table_utils.table_astype(
            table, cast_table_type, False, False
        )
        cpp_table = py_data_to_cpp_table(cast_table, (), in_col_inds, n_table_cols)
        _join_build_consume_batch(join_state, cpp_table, is_last, parallel)

    return impl


@intrinsic
def _join_probe_consume_batch(
    typingctx,
    join_state,
    cpp_table,
    kept_build_cols,
    kept_probe_cols,
    is_last,
    parallel,
):
    def codegen(context, builder, sig, args):
        out_is_last = cgutils.alloca_once(builder, lir.IntType(1))
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64).as_pointer(),
                lir.IntType(64),
                lir.IntType(64).as_pointer(),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(1).as_pointer(),
                lir.IntType(1),
            ],
        )
        kept_build_cols_arr = cgutils.create_struct_proxy(sig.args[2])(
            context, builder, value=args[2]
        )
        kept_probe_cols_arr = cgutils.create_struct_proxy(sig.args[3])(
            context, builder, value=args[3]
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="join_probe_consume_batch_py_entry"
        )
        func_args = [
            args[0],
            args[1],
            kept_build_cols_arr.data,
            kept_build_cols_arr.nitems,
            kept_probe_cols_arr.data,
            kept_probe_cols_arr.nitems,
            args[4],
            out_is_last,
            args[5],
        ]
        table_ret = builder.call(fn_tp, func_args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        items = [table_ret, builder.load(out_is_last)]
        return context.make_tuple(builder, sig.return_type, items)

    ret_type = types.Tuple([cpp_table, types.bool_])
    sig = ret_type(
        join_state, cpp_table, kept_build_cols, kept_probe_cols, is_last, parallel
    )
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def join_probe_consume_batch(
    join_state, table, is_last, used_cols=None, parallel=False
):
    """Consume a probe table batch in streaming join (probe hash table and produce
    output rows)

    Args:
        join_state (JoinState): C++ JoinState pointer
        table (table_type): probe table batch
        is_last (bool): is last batch
        used_cols (MetaType(tuple(int))): Indices of used columns in the output table.
            This should only be set by the compiler.
        parallel (bool): Is this a parallel join. This should only be set by the compiler.

    Returns:
        table_type: output table batch
    """
    in_col_inds = MetaType(join_state.probe_indices)
    n_table_cols = join_state.num_probe_arrs
    cast_table_type = join_state.key_casted_probe_table_type
    if cast_table_type == types.unknown:
        cast_table_type = table
    out_table_type = join_state.output_type

    # Determine the number of columns in the build/probe tables.
    build_table_type = join_state.build_table_type
    if build_table_type == types.unknown:
        num_build_cols = 0
    else:
        num_build_cols = len(build_table_type.arr_types)
    num_probe_cols = len(cast_table_type.arr_types)

    # Determine the live columns. TODO: add a used_cols and compute the result.
    if is_overload_none(used_cols):
        kept_build_cols = np.arange(num_build_cols, dtype=np.uint64)
        kept_probe_cols = np.arange(num_probe_cols, dtype=np.uint64)
        out_cols_arr = np.arange(len(out_table_type.arr_types), dtype=np.int64)
    else:
        # Used cols is a Meta Type with a sorted tuple of column indices.
        # TODO: Use that this is sorted?
        kept_cols = set(unwrap_typeref(used_cols).meta)

        live_col_counter = 0
        build_cols = []
        probe_cols = []
        out_cols = []

        # Determine the live build cols
        for i in range(num_build_cols):
            if i in kept_cols:
                build_cols.append(i)
                out_cols.append(live_col_counter)
                live_col_counter += 1
            else:
                out_cols.append(-1)

        # Determine the live probe cols
        for i in range(num_probe_cols):
            output_idx = i + num_build_cols
            if output_idx in kept_cols:
                probe_cols.append(i)
                out_cols.append(live_col_counter)
                live_col_counter += 1
            else:
                out_cols.append(-1)
        # Generate the arrays to include.
        kept_build_cols = np.array(build_cols, dtype=np.uint64)
        kept_probe_cols = np.array(probe_cols, dtype=np.uint64)
        out_cols_arr = np.array(out_cols, dtype=np.int64)

    def impl(
        join_state, table, is_last, used_cols=None, parallel=False
    ):  # pragma: no cover
        cast_table = bodo.utils.table_utils.table_astype(
            table, cast_table_type, False, False
        )
        cpp_table = py_data_to_cpp_table(cast_table, (), in_col_inds, n_table_cols)
        out_cpp_table, out_is_last = _join_probe_consume_batch(
            join_state, cpp_table, kept_build_cols, kept_probe_cols, is_last, parallel
        )
        out_table = cpp_table_to_py_table(out_cpp_table, out_cols_arr, out_table_type)
        delete_table(out_cpp_table)
        return out_table, out_is_last

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
