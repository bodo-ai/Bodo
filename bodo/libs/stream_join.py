# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Support for streaming join (a.k.a. vectorized join).
This file is mostly wrappers for C++ implementations.
"""
from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import llvmlite.binding as ll
import numba
import numpy as np
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.typing.templates import (
    AbstractTemplate,
    infer_global,
    signature,
)
from numba.extending import intrinsic, lower_builtin, models, register_model

import bodo
from bodo.ext import stream_join_cpp
from bodo.libs.array import (
    array_info_type,
    cpp_table_to_py_table,
    delete_table,
    py_data_to_cpp_table,
)
from bodo.libs.stream_base import StreamingStateType
from bodo.utils.transform import get_call_expr_arg
from bodo.utils.typing import (
    BodoError,
    MetaType,
    error_on_unsupported_streaming_arrays,
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

# The following are used for debugging and testing purposes only:
ll.add_symbol("join_get_op_pool_budget_bytes", stream_join_cpp.get_op_pool_budget_bytes)
ll.add_symbol("join_get_op_pool_bytes_pinned", stream_join_cpp.get_op_pool_bytes_pinned)
ll.add_symbol(
    "join_get_op_pool_bytes_allocated", stream_join_cpp.get_op_pool_bytes_allocated
)
ll.add_symbol(
    "join_get_num_partitions",
    stream_join_cpp.get_num_partitions,
)
ll.add_symbol(
    "join_get_partition_num_top_bits_by_idx",
    stream_join_cpp.get_partition_num_top_bits_by_idx,
)
ll.add_symbol(
    "join_get_partition_top_bitmask_by_idx",
    stream_join_cpp.get_partition_top_bitmask_by_idx,
)
ll.add_symbol(
    "runtime_join_filter_py_entry", stream_join_cpp.runtime_join_filter_py_entry
)

# Iterative type inference approach for streaming states:
# 1) init functions return state type with unknown table types to allow type inference
# to continue.
# 2) Functions that require table type for their output types throw NumbaError to
# restart type inference.
# 3) Functions that do not need table type simply return their output types.
# 4) Typing and lowering should be separated to make sure invalid code doesn't get
# compiled recursively before iterative type inference is done and state type is
# finalized.

# See here about the new Numba error handling:
# https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-old-style-numba-captured-errors


class JoinStateType(StreamingStateType):
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
        error_on_unsupported_streaming_arrays(build_table_type)
        error_on_unsupported_streaming_arrays(probe_table_type)

        self.build_key_inds = build_key_inds
        self.probe_key_inds = probe_key_inds
        self.build_outer = build_outer
        self.probe_outer = probe_outer
        self.build_table_type = build_table_type
        self.probe_table_type = probe_table_type
        super().__init__(
            (
                f"JoinStateType("
                f"build_keys={build_key_inds}, "
                f"probe_keys={probe_key_inds}, "
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
            self.build_outer,
            self.probe_outer,
            self.build_table_type,
            self.probe_table_type,
        )

    @property
    def n_keys(self):
        return len(self.build_key_inds)

    # Methods used to compute the information needed for _init_join_state

    @staticmethod
    def _derive_cpp_indices(key_indices, num_cols):
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

    @staticmethod
    def _input_table_logical_to_physical_map(key_indices, table_type) -> Dict[int, int]:
        """Return a dictionary mapping a logical Python index to a
        physical C++ index for the given input table.

        Args:
            key_indices (N Tuple(int)): The indices of the key columns
            table_type (TableType): The input table type.

        Returns:
            Dict[int, int]: Dictionary mapping logical to physical indices.
        """
        index_map = {}
        # In C++ the keys have been moved to the front of the table
        key_map = {idx: i for i, idx in enumerate(key_indices)}
        data_offset = len(key_indices)
        num_cols = len(table_type.arr_types)
        for i in range(num_cols):
            if i in key_map:
                physical_index = key_map[i]
            else:
                physical_index = data_offset
                data_offset += 1
            index_map[i] = physical_index
        return index_map

    def build_logical_to_physical_map(self) -> Dict[int, int]:
        """Return a dictionary mapping a logical Python index to a
        physical C++ index for the build table. This assumes all
        columns are live and is used for non-equality joins.

        Returns:
            Dict[int, int]: Dictionary mapping logical to physical indices.
        """
        return self._input_table_logical_to_physical_map(
            self.build_key_inds, self.build_table_type
        )

    def probe_logical_to_physical_map(self) -> Dict[int, int]:
        """Return a dictionary mapping a logical Python index to a
        physical C++ index for the probe table. This assumes all
        columns are live and is used for non-equality joins.

        Returns:
            Dict[int, int]: Dictionary mapping logical to physical indices.
        """
        return self._input_table_logical_to_physical_map(
            self.probe_key_inds, self.probe_table_type
        )

    @staticmethod
    def _derive_input_type(
        key_types, key_indices, table_type
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

    @staticmethod
    def _derive_common_key_type(input_types: list[types.ArrayCompatible]):
        """
        Derive a common key type for a list of array input types.
        This is used for unifying the build/probe key types and for
        unifying the key types for duplicated columns.

        Args:
            input_types (List[types.ArrayCompatible]): List of array types
            to unify.

        Raises:
            BodoError: The keys cannot be unified.

        Returns:
            types.ArrayCompatible: The final output array type.
        """
        assert len(input_types) > 0, "At least 1 key type must be provided"
        are_nullable = [is_nullable(t) for t in input_types]
        # Make sure arrays have the same nullability.
        if any([nullable for nullable in are_nullable]):
            input_types = [to_nullable_type(t) for t in input_types]

        # Check if all array types are the same.
        if all([t == input_types[0] for t in input_types]):
            return input_types[0]

        if isinstance(input_types[0], bodo.MapArrayType):
            assert all(
                [isinstance(t, bodo.MapArrayType) for t in input_types]
            ), f"StreamingHashJoin: Cannot unify Map array with non-Map arrays! {input_types=}"
            common_key_arr_type = JoinStateType._derive_common_key_type(
                [t.key_arr_type for t in input_types]
            )
            common_value_arr_types = JoinStateType._derive_common_key_type(
                [t.value_arr_type for t in input_types]
            )
            return bodo.MapArrayType(common_key_arr_type, common_value_arr_types)

        if isinstance(input_types[0], bodo.ArrayItemArrayType):
            assert all(
                [isinstance(t, bodo.ArrayItemArrayType) for t in input_types]
            ), f"StreamingHashJoin: Cannot unify List array with non-List arrays! {input_types=}"

            common_element_type = JoinStateType._derive_common_key_type(
                [t.dtype for t in input_types]
            )
            return bodo.ArrayItemArrayType(common_element_type)

        if isinstance(input_types[0], bodo.StructArrayType):
            assert all(
                [isinstance(t, bodo.StructArrayType) for t in input_types]
            ), f"StreamingHashJoin: Cannot unify Struct array with non-Struct arrays! {input_types=}"
            n_fields = len(input_types[0].data)
            assert all([len(t.data) == n_fields for t in input_types])
            field_names = input_types[0].names
            assert all([t.names == field_names for t in input_types])

            common_field_types = []
            for i in range(len(input_types[0].data)):
                common_field_types.append(
                    JoinStateType._derive_common_key_type(
                        [t.data[i] for t in input_types]
                    )
                )
            return bodo.StructArrayType(tuple(common_field_types), field_names)

        # Handle non-nested types:
        are_bodosql_integer_arr_types = [
            is_bodosql_integer_arr_type(t) for t in input_types
        ]
        if all(are_bodosql_integer_arr_types):
            # TODO: Future optimization. If the types don't match exactly and
            # we don't have an outer join on the larger bitwidth side, we don't
            # have to upcast and can instead filter + downcast. In particular we
            # know that any type that doesn't fit in the smaller array type
            # cannot have a match.
            #
            # Note: If we have an outer join we can't do this because we need to
            # preserve the elements without matches.
            #
            common_type = get_common_bodosql_integer_arr_type(input_types)
        else:
            # If the inputs are all string or dict, return string.
            valid_str_types = (bodo.string_array_type, bodo.dict_str_arr_type)
            if all([t in valid_str_types for t in input_types]):
                common_type = bodo.string_array_type
            else:
                raise BodoError(
                    f"StreamingHashJoin: Build and probe keys must have the same types. {input_types=}"
                )
        return common_type

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
        # Check if a key is used more than once.
        seen_build_keys = defaultdict(int)
        seen_probe_keys = defaultdict(int)
        for i in range(num_keys):
            build_key_index = build_key_inds[i]
            probe_key_index = probe_key_inds[i]
            seen_build_keys[build_key_index] += 1
            seen_probe_keys[probe_key_index] += 1
            build_arr_type = self.build_table_type.arr_types[build_key_index]
            probe_arr_type = self.probe_table_type.arr_types[probe_key_index]
            key_type = self._derive_common_key_type([build_arr_type, probe_arr_type])
            arr_types.append(key_type)

        # Check if a key is used more than once. If so we need to
        # unify those keys.
        if len(seen_build_keys) != num_keys or len(seen_probe_keys) != num_keys:
            # Find the key indices that need to match
            kept_build_keys = set()
            kept_probe_keys = set()
            for key, val in seen_build_keys.items():
                if val > 1:
                    kept_build_keys.add(key)
            for key, val in seen_probe_keys.items():
                if val > 1:
                    kept_probe_keys.add(key)
            # Find the indices of those keys
            indices_to_match = []
            for i in range(num_keys):
                build_key_index = build_key_inds[i]
                probe_key_index = probe_key_inds[i]
                if (
                    build_key_index in kept_build_keys
                    or probe_key_index in kept_probe_keys
                ):
                    indices_to_match.append(i)
            common_key_type = self._derive_common_key_type(
                [arr_types[i] for i in indices_to_match]
            )
            # Update the key types
            for i in indices_to_match:
                arr_types[i] = common_key_type

        return arr_types

    @staticmethod
    def _key_casted_table_type(key_types, key_indices, table_type):
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
        keys_map = {idx: key_types[i] for i, idx in enumerate(key_indices)}
        arr_types = []
        for i in range(len(table_type.arr_types)):
            if i in keys_map:
                arr_types.append(keys_map[i])
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
    def num_build_input_arrs(self) -> int:
        """
        Determine the actual number of build arrays in the input.

        Note: We use build_reordered_arr_types in case the same column
        is used as a key in multiple comparisons.

        Return (int): The number of build arrays
        """
        return len(self.build_reordered_arr_types)

    @property
    def num_probe_input_arrs(self) -> int:
        """
        Determine the actual number of probe arrays in the input.

        Note: We use probe_reordered_arr_types in case the same column
        is used as a key in multiple comparisons.

        Return (int): The number of probe arrays
        """
        return len(self.probe_reordered_arr_types)

    def _compute_table_output_arrays(
        self, key_indices, table_type, make_nullable: bool
    ):
        """Compute one side of the output arrays for the output table.

        Args:
            key_indices (N Tuple(int)): The indices of the key columns
            table_type (TableType): The input table type for that side.
            make_nullable (bool): Should the output be converted to nullable types.

        Returns:
            List[types.ArrayCompatible]: The list of output array types.
        """
        arr_types = []
        # Note: We can maintain the input key types in the output.
        # Nothing should assume the original type if we cast an integer.
        key_types = self.key_types
        key_map = {key: key_types[i] for i, key in enumerate(key_indices)}
        for i in range(len(table_type.arr_types)):
            if i in key_map:
                arr_type = key_map[i]
            else:
                arr_type = table_type.arr_types[i]
            if make_nullable:
                arr_type = to_nullable_type(arr_type)
            arr_types.append(arr_type)
        return arr_types

    @property
    def build_output_arrays(self) -> List[types.ArrayCompatible]:
        """Determine the output array types in the correct order for
        the build side. This is used to generate the output type.

        Returns:
            List[types.ArrayCompatible]: The list of output array types.
        """
        return self._compute_table_output_arrays(
            self.build_key_inds, self.build_table_type, self.probe_outer
        )

    @property
    def probe_output_arrays(self) -> List[types.ArrayCompatible]:
        """Determine the output array types in the correct order for
        the probe side. This is used to generate the output type.

        Returns:
            List[types.ArrayCompatible]: The list of output array types.
        """
        return self._compute_table_output_arrays(
            self.probe_key_inds, self.probe_table_type, self.build_outer
        )

    @property
    def output_type(self):
        """Return the output type from generating this join. BodoSQL
        expects the output type to have the same columns as the input
        types and in the same locations. This means keys may not necessarily
        be in the front.

        Returns:
            bodo.TableType: The type of the output table.
        """
        arr_types = self.probe_output_arrays + self.build_output_arrays
        out_table_type = bodo.TableType(tuple(arr_types))
        return out_table_type

    def _get_table_live_col_arrs(
        self,
        key_indices,
        table_type,
        kept_cols: Set[int],
        live_col_offset: int,
        kept_check_offset: int,
    ) -> Tuple[List[int], List[int], int]:
        """Compute the column indices for telling C++ which columns to keep live in
        the output and where to find each column for each input table. This returns
        3 values, two of which are lists. The first list will be passed to RetrieveTable
        in C++ to prune/reorder columns to match the output type for the column. The second
        list will then match each column to its C++ location (or -1 if it is dead).

        The general idea is since key columns are moved to the front for compute and potentially
        duplicated, the first list will reorder the columns back to map the original type
        (except for any dead columns) so the second list only has to determine if a column is
        live (and how many prior live columns exist) and can be filled in type order.

        Args:
            key_indices (N Tuple(int)): The indices of the key columns
            table_type (TableType): The input table type for that side
            kept_cols (Set[int]): Set of indices for which columns are live in the output
                table.
            live_col_offset (int): Offset to add to the number of live columns.
            kept_check_offset (int): Offset to apply to a column index to check
                if it is in kept_cols.

        Returns:
            Tuple[List[int], List[int], int]: Returns a tuple of 3 values:
                - The list of C++ column indices for the table that should be included
                  in the output table. This may reorder or drop columns.
                - The list of C++ column indices for the output table that maps each input
                  column to its C++ column. If a column is dead it will have the value of -1.
                - The number of live columns in the output table.
        """

        output_cols = []
        table_cols = []
        key_map = {idx: i for i, idx in enumerate(key_indices)}
        data_offset = self.n_keys
        num_cols = len(table_type.arr_types)
        live_col_counter = live_col_offset
        # Determine the live cols
        for i in range(num_cols):
            # Find the physical index of the column
            # + update the data offset if it is not a key
            if i in key_map:
                physical_index = key_map[i]
            else:
                physical_index = data_offset
                data_offset += 1
            # Update the results based on if the column is live.
            checked_index = i + kept_check_offset
            if checked_index in kept_cols:
                table_cols.append(physical_index)
                output_cols.append(live_col_counter)
                live_col_counter += 1
            else:
                output_cols.append(-1)
        return (
            table_cols,
            output_cols,
            live_col_counter,
        )

    def get_output_live_col_arrs(
        self, used_cols
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate Numpy arrays of indices to lower to the runtime indicating
        which columns should be live from the output of build, probe, and the overall
        output type. The build and probe arrays are passed to RetrieveTable to
        avoid generating the output columns and reorder columns so they match
        the expected output type.

        Args:
            used_cols (Union[MetaType, types.None]): The used cols value
                passed to the join to prune unused columns. If none all columns
                in the original output type should be used. Otherwise the metatype
                contains the live columns.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Returns a tuple of 3 values:
                - The array of C++ column indices for the probe table that should be included
                  in the output table. This may reorder or drop columns.
                - The array of C++ column indices for the build table that should be included
                  in the output table. This may reorder or drop columns.
                - The array of C++ column indices for the output table that maps each input
                  column to its C++ column. If a column is dead it will have the value of -1.
        """
        if (
            self.build_table_type == types.unknown
            or self.probe_table_type == types.unknown
        ):
            # Only compute when the types are final.
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )
        out_table_type = self.output_type
        num_output_cols = len(out_table_type.arr_types)
        if is_overload_none(used_cols):
            # All of the original inputs are kept
            kept_cols = set(range(num_output_cols))
        else:
            # Used cols is a Meta Type with a sorted tuple of column indices.
            kept_cols = set(unwrap_typeref(used_cols).meta)

        out_cols = []
        # Compute the probe side
        probe_cols, probe_output, live_col_counter = self._get_table_live_col_arrs(
            self.probe_key_inds, self.probe_table_type, kept_cols, 0, 0
        )
        # Compute the build side
        build_cols, build_output, _ = self._get_table_live_col_arrs(
            self.build_key_inds,
            self.build_table_type,
            kept_cols,
            live_col_counter,
            len(self.probe_table_type.arr_types),
        )
        out_cols = probe_output + build_output

        # Return the results as arrays
        return (
            np.array(probe_cols, dtype=np.int64),
            np.array(build_cols, dtype=np.int64),
            np.array(out_cols, dtype=np.int64),
        )


register_model(JoinStateType)(models.OpaqueModel)


@intrinsic
def _init_join_state(
    typingctx,
    operator_id,
    build_arr_dtypes,
    build_arr_array_types,
    n_build_arrs,
    probe_arr_dtypes,
    probe_arr_array_types,
    n_probe_arrs,
    interval_build_columns_arr,
    n_interval_build_columns,
    force_broadcast_t,
    op_pool_size_bytes_t,
    output_state_type,
    cfunc_cond_t,
    build_parallel_t,
    probe_parallel_t,
):
    """Initialize C++ JoinState pointer

    Args:
        operator_id (int64): ID of this operator (used for looking up budget),
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
        interval_build_columns_arr (int64*): pointer to array of ints representing the build indices for interval joins
        n_interval_build_columns (int64): number of build columns for interval joins
        force_broadcast_t (bool): Should we broadcast the build side regardless of size.
        op_pool_size_bytes_t (int64): Number of pinned bytes that this operator is allowed
            to use. Set this to -1 to let the operator use a pre-determined portion of
            the total available memory.
        output_state_type (TypeRef[JoinStateType]): The output type for the state that should be
                                                    generated.
        cfunc_cond_t (void *): Non-equality condition function. Nullptr if the join only has equality
                            conditions or is a true cross join.
        build_parallel_t (bool): Is the build table parallel?
        probe_parallel_t (bool): Is the probe table parallel?
    """
    output_type = unwrap_typeref(output_state_type)

    def codegen(context, builder, sig, args):
        (
            operator_id,
            build_arr_dtypes,
            build_arr_array_types,
            n_build_arrs,
            probe_arr_dtypes,
            probe_arr_array_types,
            n_probe_arrs,
            interval_build_columns_arr,
            n_interval_build_columns,
            force_broadcast,
            op_pool_size_bytes,
            _,
            cfunc_cond,
            build_parallel,
            probe_parallel,
        ) = args
        n_keys = context.get_constant(types.uint64, output_type.n_keys)
        build_table_outer = context.get_constant(types.bool_, output_type.build_outer)
        probe_table_outer = context.get_constant(types.bool_, output_type.probe_outer)
        output_batch_size = context.get_constant(
            types.int64, bodo.bodosql_streaming_batch_size
        )
        sync_iter = context.get_constant(types.int64, bodo.stream_loop_sync_iters)
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(64),
                lir.IntType(64).as_pointer(),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="join_state_init_py_entry"
        )
        input_args = (
            operator_id,
            build_arr_dtypes,
            build_arr_array_types,
            n_build_arrs,
            probe_arr_dtypes,
            probe_arr_array_types,
            n_probe_arrs,
            n_keys,
            interval_build_columns_arr,
            n_interval_build_columns,
            build_table_outer,
            probe_table_outer,
            force_broadcast,
            cfunc_cond,
            build_parallel,
            probe_parallel,
            output_batch_size,
            sync_iter,
            op_pool_size_bytes,
        )
        ret = builder.call(fn_tp, input_args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = output_type(
        types.int64,
        types.voidptr,
        types.voidptr,
        types.int32,
        types.voidptr,
        types.voidptr,
        types.int32,
        types.CPointer(types.int64),
        types.int64,
        types.bool_,
        types.int64,  # op_pool_size_bytes_t
        output_state_type,
        types.voidptr,
        build_parallel_t,
        probe_parallel_t,
    )
    return sig, codegen


def init_join_state(
    operator_id,
    build_key_inds,
    probe_key_inds,
    build_colnames,
    probe_colnames,
    build_outer,
    probe_outer,
    interval_build_columns,
    force_broadcast,
    op_pool_size_bytes=-1,
    expected_state_type=None,
    # The non-equality portion of the join condition. If None then
    # the join is a pure hash join. Otherwise this is a string similar
    # to the query string accepted by merge.
    non_equi_condition=None,
    # Note: build_parallel and probe_parallel are set automatically
    # by the compiler and should not be set by the user.
    build_parallel=False,
    probe_parallel=False,
):
    pass


def _get_init_join_state_type(
    expected_state_type,
    build_key_inds,
    probe_key_inds,
    build_outer,
    probe_outer,
):
    """Helper for init_join_state output typing that returns state type with unknown
    table types when expected state type is not provided.
    """
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

    return output_type


@infer_global(init_join_state)
class InitJoinStateInfer(AbstractTemplate):
    """Typer for init_join_state that returns join state type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(init_join_state)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        expected_state_type = unwrap_typeref(folded_args[10])
        (
            build_key_inds,
            probe_key_inds,
            _,
            _,
            build_outer,
            probe_outer,
        ) = folded_args[1:7]
        output_type = _get_init_join_state_type(
            expected_state_type,
            build_key_inds,
            probe_key_inds,
            build_outer,
            probe_outer,
        )
        return signature(output_type, *folded_args).replace(pysig=pysig)


InitJoinStateInfer._no_unliteral = True


@lower_builtin(init_join_state, types.VarArg(types.Any))
def lower_init_join_state(context, builder, sig, args):
    """lower init_join_state() using gen_init_join_state_impl"""
    impl = gen_init_join_state_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def gen_init_join_state_impl(
    operator_id,
    build_key_inds,
    probe_key_inds,
    build_colnames,
    probe_colnames,
    build_outer,
    probe_outer,
    interval_build_columns,
    force_broadcast,
    op_pool_size_bytes=-1,
    expected_state_type=None,
    # The non-equality portion of the join condition. If None then
    # the join is a pure hash join. Otherwise this is a string similar
    # to the query string accepted by merge.
    non_equi_condition=None,
    # Note: build_parallel and probe_parallel are set automatically
    # by the compiler and should not be set by the user.
    build_parallel=False,
    probe_parallel=False,
):
    output_type = unwrap_typeref(expected_state_type)

    build_arr_dtypes = output_type.build_arr_ctypes
    build_arr_array_types = output_type.build_arr_array_types
    n_build_arrs = len(build_arr_array_types)
    probe_arr_dtypes = output_type.probe_arr_ctypes
    probe_arr_array_types = output_type.probe_arr_array_types
    n_probe_arrs = len(probe_arr_array_types)
    interval_build_columns = unwrap_typeref(interval_build_columns).meta
    build_indices = output_type.build_indices
    updated_interval_build_columns = []
    for build_idx in interval_build_columns:
        # Before we have the expected state type the mapping won't work.
        if build_idx < len(build_indices):
            updated_interval_build_columns.append(build_indices[build_idx])
    updated_interval_build_columns_arr = np.array(
        updated_interval_build_columns, dtype=np.int64
    )
    num_interval_build_cols = len(updated_interval_build_columns_arr)

    # handle non-equi conditions (reuse existing join code as much as possible)
    # Note we must account for how keys will be cast here.
    build_table_type = output_type.key_casted_build_table_type
    probe_table_type = output_type.key_casted_probe_table_type

    if not is_overload_none(non_equi_condition):
        from bodo.ir.join import (
            add_join_gen_cond_cfunc_sym,
            gen_general_cond_cfunc,
            get_join_cond_addr,
        )

        gen_expr_const = get_overload_const_str(non_equi_condition)

        build_column_names = unwrap_typeref(build_colnames).meta
        probe_column_names = unwrap_typeref(probe_colnames).meta
        # Parse the query
        _, _, parsed_gen_expr = bodo.hiframes.dataframe_impl._parse_merge_cond(
            gen_expr_const,
            probe_column_names,
            probe_table_type.arr_types,
            build_column_names,
            build_table_type.arr_types,
        )
        left_logical_to_physical = output_type.probe_logical_to_physical_map()
        right_logical_to_physical = output_type.build_logical_to_physical_map()
        left_var_map = {c: i for i, c in enumerate(probe_column_names)}
        right_var_map = {c: i for i, c in enumerate(build_column_names)}

        # Generate a general join condition cfunc
        general_cond_cfunc, _, _ = gen_general_cond_cfunc(
            None,
            left_logical_to_physical,
            right_logical_to_physical,
            str(parsed_gen_expr),
            left_var_map,
            None,
            set(),
            probe_table_type,
            right_var_map,
            None,
            set(),
            build_table_type,
            compute_in_batch=not output_type.build_key_inds,
        )
        cfunc_native_name = general_cond_cfunc.native_name

        def impl_nonequi(
            operator_id,
            build_key_inds,
            probe_key_inds,
            build_colnames,
            probe_colnames,
            build_outer,
            probe_outer,
            interval_build_columns,
            force_broadcast,
            op_pool_size_bytes=-1,
            expected_state_type=None,
            non_equi_condition=None,
            build_parallel=False,
            probe_parallel=False,
        ):  # pragma: no cover
            cfunc_cond = add_join_gen_cond_cfunc_sym(
                general_cond_cfunc, cfunc_native_name
            )
            cfunc_cond = get_join_cond_addr(cfunc_native_name)
            return _init_join_state(
                operator_id,
                build_arr_dtypes.ctypes,
                build_arr_array_types.ctypes,
                n_build_arrs,
                probe_arr_dtypes.ctypes,
                probe_arr_array_types.ctypes,
                n_probe_arrs,
                updated_interval_build_columns_arr.ctypes,
                num_interval_build_cols,
                force_broadcast,
                op_pool_size_bytes,
                output_type,
                cfunc_cond,
                build_parallel,
                probe_parallel,
            )

        return impl_nonequi

    def impl_init_join_state(
        operator_id,
        build_key_inds,
        probe_key_inds,
        build_colnames,
        probe_colnames,
        build_outer,
        probe_outer,
        interval_build_columns,
        force_broadcast,
        op_pool_size_bytes=-1,
        expected_state_type=None,
        non_equi_condition=None,
        build_parallel=False,
        probe_parallel=False,
    ):  # pragma: no cover
        return _init_join_state(
            operator_id,
            build_arr_dtypes.ctypes,
            build_arr_array_types.ctypes,
            n_build_arrs,
            probe_arr_dtypes.ctypes,
            probe_arr_array_types.ctypes,
            n_probe_arrs,
            updated_interval_build_columns_arr.ctypes,
            num_interval_build_cols,
            force_broadcast,
            op_pool_size_bytes,
            output_type,
            0,
            build_parallel,
            probe_parallel,
        )

    return impl_init_join_state


@intrinsic
def _join_build_consume_batch(
    typingctx,
    join_state,
    cpp_table,
    is_last,
):
    def codegen(context, builder, sig, args):
        request_input = cgutils.alloca_once(builder, lir.IntType(1))
        fnty = lir.FunctionType(
            lir.IntType(1),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="join_build_consume_batch_py_entry"
        )
        ret = builder.call(fn_tp, tuple(args) + (request_input,))
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return context.make_tuple(
            builder, sig.return_type, [ret, builder.load(request_input)]
        )

    ret_type = types.Tuple([types.bool_, types.bool_])
    sig = ret_type(join_state, cpp_table, is_last)
    return sig, codegen


def join_build_consume_batch(join_state, table, is_last):
    pass


def gen_join_build_consume_batch_impl(join_state, table, is_last):
    """Consume a build table batch in streaming join (insert into hash table)

    Args:
        join_state (JoinState): C++ JoinState pointer
        table (table_type): build table batch
        is_last (bool): is last batch locally
    Returns:
        tuple(bool, bool): is last batch globally with possibility of false negatives
         due to iterations between syncs, whether to request input rows from preceding
         operators
    """
    in_col_inds = MetaType(join_state.build_indices)
    n_table_cols = join_state.num_build_input_arrs
    cast_table_type = join_state.key_casted_build_table_type

    def impl_join_build_consume_batch(join_state, table, is_last):  # pragma: no cover
        cast_table = bodo.utils.table_utils.table_astype(
            table, cast_table_type, False, False
        )
        cpp_table = py_data_to_cpp_table(cast_table, (), in_col_inds, n_table_cols)
        return _join_build_consume_batch(join_state, cpp_table, is_last)

    return impl_join_build_consume_batch


@infer_global(join_build_consume_batch)
class JoinBuildConsumeBatchInfer(AbstractTemplate):
    """Typer for join_build_consume_batch that returns bool as output type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(join_build_consume_batch)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        output_type = types.BaseTuple.from_types((types.bool_, types.bool_))
        return signature(output_type, *folded_args).replace(pysig=pysig)


JoinBuildConsumeBatchInfer._no_unliteral = True


@lower_builtin(join_build_consume_batch, types.VarArg(types.Any))
def lower_join_build_consume_batch(context, builder, sig, args):
    """lower join_build_consume_batch() using gen_join_build_consume_batch_impl above"""
    impl = gen_join_build_consume_batch_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


@intrinsic
def _runtime_join_filter(
    typingctx,
    join_state,
    cpp_table,
    row_bitmask,
    join_key_idxs,
    process_col_bitmask,
    applied_any_filter,
):
    def codegen(context, builder, sig, args):
        join_key_idxs_arr = cgutils.create_struct_proxy(sig.args[3])(
            context, builder, value=args[3]
        )
        process_col_bitmask_arr = cgutils.create_struct_proxy(sig.args[4])(
            context, builder, value=args[4]
        )
        fnty = lir.FunctionType(
            lir.IntType(1),
            [
                lir.IntType(8).as_pointer(),  # join_state
                lir.IntType(8).as_pointer(),  # cpp_table
                lir.IntType(8).as_pointer(),  # bitmask
                lir.IntType(64).as_pointer(),  # join_key_idxs
                lir.IntType(64),  # join_key_idxs_len
                lir.IntType(8).as_pointer(),  # process_col_bitmask
                lir.IntType(64),  # process_col_bitmask_len
                lir.IntType(1),  # applied_any_filter
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="runtime_join_filter_py_entry"
        )
        func_args = [
            args[0],
            args[1],
            args[2],
            join_key_idxs_arr.data,
            join_key_idxs_arr.nitems,
            process_col_bitmask_arr.data,
            process_col_bitmask_arr.nitems,
            args[5],
        ]
        applied_any_filter = builder.call(fn_tp, func_args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return applied_any_filter

    sig = types.bool_(
        join_state,
        cpp_table,
        array_info_type,
        join_key_idxs,
        process_col_bitmask,
        types.bool_,
    )

    return sig, codegen


def runtime_join_filter(
    join_states,
    table,
    join_keys_idxs,
    process_col_bitmasks,
):
    """
    Apply one or more runtime join filters to a table.

    Args:
        join_state (Metatype(tuple[JoinState]): Tuple of C++ JoinState pointer
        table (table_type): Table to filter.
        join_key_idxs (Metatype(tuple[Metatype(tuple[int])])): Tuple of tuples where
            the k'th tuple has length join_state[k]->n_keys specifying the index
            of the column in the input table that corresponds
            to the i'th join key. -1 if no column corresponding
            to the i'th join key exists. If none of these are -1,
            a bloom filter will be applied.
        process_col_bitmask (Metatype(tuple[Metatype(tuple[bool])])): Tuple of tuples where
            the k'th tuple join_state[k]->n_keys specifying whether or not to apply
            a column level filter on the column corresponding to
            the i'th join key.
    """
    pass


def _get_runtime_join_filter_info(
    join_states, input_table_t, join_keys_idxs, process_col_bitmasks
):
    """Compute typing information for runtime join filter to use during typing and
    lowering codegen.
    """
    num_filters = len(join_states)
    n_cols = len(input_table_t.arr_types)

    join_key_idxs_lists = tuple(
        list(unwrap_typeref(join_key_idxs).meta) for join_key_idxs in join_keys_idxs
    )
    process_col_bitmask_lists = tuple(
        list(unwrap_typeref(process_col_bitmask).meta)
        for process_col_bitmask in process_col_bitmasks
    )

    input_idx_to_join_key_idxs = tuple(
        {
            join_key_idxs_list[i]: i
            for i in range(len(join_key_idxs_list))
            if join_key_idxs_list[i] != -1
        }
        for join_key_idxs_list in join_key_idxs_lists
    )

    # Bloom filter can only be applied when all key columns are present
    can_apply_bloom_filters = [
        len(join_key_idxs_list) > 0 and all([(idx != -1) for idx in join_key_idxs_list])
        for join_key_idxs_list in join_key_idxs_lists
    ]

    # At this time, the only column level filter we support is when both
    # the input and its corresponding join key are DICT arrays. If we see
    # such a combination and if process_col_bitmask_list[i] is true for that
    # column, we will set this to True.
    can_apply_col_filters = [False for _ in range(num_filters)]

    cast_table_types = []

    # Cast the key columns to the correct types (e.g. int32 to int64).
    # The output type will be the type after the cast to avoid
    # casting back to the original type. This is fine since we would
    # do this cast at some point anyway.
    for i in range(num_filters):
        # NOTE: key_types may not be available during type inference, so we just assume
        # no cast is necessary and return the input table type to allow type inference
        # to continue.
        join_key_types = join_states[i].key_types
        if join_key_types == []:
            cast_table_types.append(input_table_t)
        else:
            cast_arr_types = []
            valid_str_types = (bodo.string_array_type, bodo.dict_str_arr_type)
            for j in range(n_cols):
                if j in input_idx_to_join_key_idxs[i]:
                    # If this is a key column, cast it to the join key type.
                    # In case the key type is DICT and the corresponding input column
                    # is STRING, we will keep it as STRING. The C++ function won't
                    # do column level filtering on the column, but it will be able to
                    # use the string values directly for a bloom filter check.
                    # If the key type is STRING and the corresponding input column is
                    # DICT, we will convert the input to a STRING. This will allow us to
                    # apply the bloom filter.
                    input_col_t = input_table_t.arr_types[j]
                    join_key_t = join_key_types[input_idx_to_join_key_idxs[i][j]]
                    if (
                        (input_col_t != join_key_t)
                        and (input_col_t in valid_str_types)
                        and (join_key_t in valid_str_types)
                    ):
                        casted_type = bodo.string_array_type
                    else:
                        casted_type = join_key_types[input_idx_to_join_key_idxs[i][j]]
                    cast_arr_types.append(casted_type)

                    # At this point, this is the only case where a column level filter
                    # would be applied.
                    if (
                        casted_type == bodo.dict_str_arr_type
                        and join_key_t == bodo.dict_str_arr_type
                        and process_col_bitmask_lists[i][
                            input_idx_to_join_key_idxs[i][j]
                        ]
                        == True
                    ):
                        can_apply_col_filters[i] = True
                else:
                    # If not a key column, then preserve the original type
                    cast_arr_types.append(input_table_t.arr_types[j])

            cast_table_types.append(bodo.TableType(tuple(cast_arr_types)))
        cast_table_types_tuple = tuple(cast_table_types)

    return (
        cast_table_types_tuple,
        can_apply_bloom_filters,
        can_apply_col_filters,
        join_key_idxs_lists,
    )


@infer_global(runtime_join_filter)
class RuntimeJoinFilterInfer(AbstractTemplate):
    """Typer for runtime_join_filter that returns output table type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(runtime_join_filter)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        join_states, table, join_keys_idxs, process_col_bitmasks = folded_args
        # NOTE: join state table types may not be available during type inference, but
        # we return input types without casting to allow type inference to continue.
        # Otherwise, type inference may not reach probe calls to type join states
        # properly. This is correct since type inference will refine the types later.

        (
            cast_table_types_tuple,
            can_apply_bloom_filters,
            can_apply_col_filters,
            _,
        ) = _get_runtime_join_filter_info(
            join_states, table, join_keys_idxs, process_col_bitmasks
        )

        # Return input type if no filter can be applied
        if (not any(can_apply_bloom_filters)) and (not any(can_apply_col_filters)):
            return signature(table, *folded_args).replace(pysig=pysig)

        # the index of the last table created
        curr_table_idx = -1
        # assume at least one filter can be applied (so cpp_table is always defined)
        num_filters = len(join_states)
        for i in range(num_filters):
            if can_apply_bloom_filters[i] or can_apply_col_filters[i]:
                curr_table_idx = i

        return signature(cast_table_types_tuple[curr_table_idx], *folded_args).replace(
            pysig=pysig
        )


RuntimeJoinFilterInfer._no_unliteral = True


@lower_builtin(runtime_join_filter, types.VarArg(types.Any))
def lower_runtime_join_filter(context, builder, sig, args):
    """lower runtime_join_filter() using overload_runtime_join_filter"""
    impl = overload_runtime_join_filter(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def overload_runtime_join_filter(
    join_states, table, join_keys_idxs, process_col_bitmasks
):
    num_filters = len(join_states)

    if any(join_state.probe_outer for join_state in join_states):
        raise BodoError("Cannot apply runtime join filter in the probe_outer case!")

    n_cols = len(table.arr_types)
    input_table_t = table

    # count the number of variable length columns to determine if cost of
    # materializing the table is high enough to justify bitmasks
    num_var_type_columns = 0
    for arr_type in input_table_t.arr_types:
        if (
            arr_type == bodo.string_array_type
            or arr_type == bodo.binary_array_type
            or isinstance(
                arr_type,
                (
                    bodo.MapArrayType,
                    bodo.ArrayItemArrayType,
                ),
            )
        ):
            num_var_type_columns += 1

    # whether to materialize after every filter or wait until the end
    materialize_after_each_filter = num_var_type_columns < int(
        bodo.runtime_join_filters_copy_threshold
    )

    (
        cast_table_types_tuple,
        can_apply_bloom_filters,
        can_apply_col_filters,
        join_key_idxs_lists,
    ) = _get_runtime_join_filter_info(
        join_states, input_table_t, join_keys_idxs, process_col_bitmasks
    )

    # If no filter can be applied, just return a NOP implementation
    # (that will get optimized out) to avoid function call overheads.
    if (not any(can_apply_bloom_filters)) and (not any(can_apply_col_filters)):
        return (
            lambda join_states, table, join_keys_idxs, process_col_bitmasks: table
        )  # pragma: no cover

    col_inds_t = MetaType(tuple(range(n_cols)))
    col_ind_arr = np.arange(n_cols)
    join_key_idxs_arrs = tuple(
        np.array(join_key_idxs_list) for join_key_idxs_list in join_key_idxs_lists
    )
    process_col_bitmask_arrs = tuple(
        np.array(unwrap_typeref(process_col_bitmask).meta)
        for process_col_bitmask in process_col_bitmasks
    )

    func_text = f"""
def impl_runtime_join_filter(
    join_states, table, join_keys_idxs, process_col_bitmasks\n
):
    row_bitmask = bodo.libs.bool_arr_ext.alloc_true_bool_array(len(table))
    applied_any_filter = False\n
    cast_table = table\n
"""

    # the index of the last table created
    curr_table_idx = -1
    # assume at least one filter can be applied (so cpp_table is always defined)
    for i in range(num_filters):
        if can_apply_bloom_filters[i] or can_apply_col_filters[i]:
            # doing repeated casting could potentially become expensive
            func_text += f"    cast_table = bodo.utils.table_utils.table_astype(\n"
            func_text += f"        cast_table, cast_table_types[{i}], False, False\n"
            func_text += f"    )\n"
            func_text += f"    cpp_table = bodo.libs.array.py_data_to_cpp_table(cast_table, (), col_inds_t, {n_cols})\n"
            func_text += (
                f"    row_bitmask_arr = bodo.libs.array.array_to_info(row_bitmask)\n"
            )
            func_text += f"    applied_any_filter = _runtime_join_filter(\n"
            func_text += f"        join_states[{i}],\n"
            func_text += f"        cpp_table,\n"
            func_text += f"        row_bitmask_arr,\n"
            func_text += f"        join_key_idxs_arrs[{i}],\n"
            func_text += f"        process_col_bitmask_arrs[{i}],\n"
            func_text += f"        applied_any_filter,\n"
            func_text += f"    )\n"

            if materialize_after_each_filter:
                func_text += f"    if applied_any_filter:\n"
                func_text += f"        row_bitmask_arr = bodo.libs.array.array_to_info(row_bitmask)\n"
                func_text += f"        cpp_table = bodo.libs.array.py_data_to_cpp_table(cast_table, (), col_inds_t, {n_cols})\n"
                func_text += f"        cpp_table = bodo.libs.array._retrieve_table(\n"
                func_text += f"            cpp_table, row_bitmask_arr\n"
                func_text += f"        )\n"
                func_text += (
                    f"        cast_table = bodo.libs.array.cpp_table_to_py_table(\n"
                )
                func_text += (
                    f"            cpp_table, col_ind_arr, cast_table_types[{i}], 0\n"
                )
                func_text += f"        )\n"
                func_text += f"        bodo.libs.array.delete_table(cpp_table)\n"
                # reallocate the bitmask for the new table, reset applied_any_filter flag
                func_text += f"        row_bitmask = bodo.libs.bool_arr_ext.alloc_true_bool_array(len(cast_table))\n"
                func_text += f"        applied_any_filter = False\n"
            curr_table_idx = i

    # if we materialize after every step, then there is no need to materialize at the end
    if materialize_after_each_filter:
        func_text += f"    return cast_table\n"
    else:
        # if rows are filtered, do table materialization
        # otherwise just return the table
        func_text += f"""
    cpp_table = bodo.libs.array.py_data_to_cpp_table(cast_table, (), col_inds_t, {n_cols})
    row_bitmask_arr = bodo.libs.array.array_to_info(row_bitmask)

    if applied_any_filter:
        out_cpp_table = bodo.libs.array._retrieve_table(
            cpp_table, row_bitmask_arr
        )
        out_table = bodo.libs.array.cpp_table_to_py_table(
            out_cpp_table, col_ind_arr, cast_table_types[{curr_table_idx}], 0
        )
        bodo.libs.array.delete_table(out_cpp_table)
    else:
        out_table = bodo.libs.array.cpp_table_to_py_table(
        cpp_table, col_ind_arr, cast_table_types[{curr_table_idx}], 0
        )
        bodo.libs.array.delete_table(cpp_table)
        bodo.libs.array.delete_info(row_bitmask_arr)

    return out_table
"""

    loc_vars = {}
    global_vars = {
        "bodo": bodo,
        "MetaType": MetaType,
        "cast_table_types": cast_table_types_tuple,
        "input_table_t": input_table_t,
        "col_ind_arr": col_ind_arr,
        "col_inds_t": col_inds_t,
        "join_key_idxs_arrs": join_key_idxs_arrs,
        "process_col_bitmask_arrs": process_col_bitmask_arrs,
        "can_apply_bloom_filters": can_apply_bloom_filters,
        "can_apply_column_filters": can_apply_col_filters,
        "_runtime_join_filter": _runtime_join_filter,
    }
    exec(func_text, global_vars, loc_vars)

    impl_runtime_join_filter = loc_vars["impl_runtime_join_filter"]

    return impl_runtime_join_filter


@intrinsic
def _join_probe_consume_batch(
    typingctx,
    join_state,
    cpp_table,
    kept_build_cols,
    kept_probe_cols,
    total_rows,
    is_last,
    produce_output,
):
    def codegen(context, builder, sig, args):
        out_is_last = cgutils.alloca_once(builder, lir.IntType(1))
        request_input = cgutils.alloca_once(builder, lir.IntType(1))
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64).as_pointer(),
                lir.IntType(64),
                lir.IntType(64).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),  # total_rows
                lir.IntType(1),
                lir.IntType(1).as_pointer(),
                lir.IntType(1),
                lir.IntType(1).as_pointer(),
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
            args[5],
            out_is_last,
            args[6],
            request_input,
        ]
        table_ret = builder.call(fn_tp, func_args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        items = [
            table_ret,
            builder.load(out_is_last),
            builder.load(request_input),
        ]
        return context.make_tuple(builder, sig.return_type, items)

    ret_type = types.Tuple([cpp_table, types.bool_, types.bool_])
    sig = ret_type(
        join_state,
        cpp_table,
        kept_build_cols,
        kept_probe_cols,
        types.voidptr,
        is_last,
        produce_output,
    )
    return sig, codegen


def join_probe_consume_batch(
    join_state,
    table,
    is_last,
    produce_output,
    used_cols=None,
):
    pass


def gen_join_probe_consume_batch_impl(
    join_state,
    table,
    is_last,
    produce_output,
    used_cols=None,
):
    """Consume a probe table batch in streaming join (probe hash table and produce
    output rows)

    Args:
        join_state (JoinState): C++ JoinState pointer
        table (table_type): probe table batch
        is_last (bool): is last batch
        produce_output (bool): whether to produce output rows
        used_cols (MetaType(tuple(int))): Indices of used columns in the output table.
            This should only be set by the compiler.

    Returns:
        table_type: output table batch
        bool: global is last batch
        bool: whether preceding operators should produce output rows (this is only a hint, not a requirement)
    """
    in_col_inds = MetaType(join_state.probe_indices)
    n_table_cols = join_state.num_probe_input_arrs
    cast_table_type = join_state.key_casted_probe_table_type
    out_table_type = join_state.output_type

    # Determine the live columns.
    (
        kept_probe_cols,
        kept_build_cols,
        out_cols_arr,
    ) = join_state.get_output_live_col_arrs(used_cols)

    def impl_join_probe_consume_batch(
        join_state, table, is_last, produce_output, used_cols=None
    ):  # pragma: no cover
        cast_table = bodo.utils.table_utils.table_astype(
            table, cast_table_type, False, False
        )
        cpp_table = py_data_to_cpp_table(cast_table, (), in_col_inds, n_table_cols)
        # Store the total rows in the output table in case all columns are dead.
        total_rows_np = np.array([0], dtype=np.int64)
        (
            out_cpp_table,
            out_is_last,
            request_input,
        ) = _join_probe_consume_batch(
            join_state,
            cpp_table,
            kept_build_cols,
            kept_probe_cols,
            total_rows_np.ctypes,
            is_last,
            produce_output,
        )
        out_table = cpp_table_to_py_table(
            out_cpp_table, out_cols_arr, out_table_type, total_rows_np[0]
        )
        delete_table(out_cpp_table)
        return out_table, out_is_last, request_input

    return impl_join_probe_consume_batch


@infer_global(join_probe_consume_batch)
class JoinProbeConsumeBatchInfer(AbstractTemplate):
    """Typer for join_probe_consume_batch that returns (output_table_type, bool, bool)
    as output type.
    """

    def generic(self, args, kws):
        kws = dict(kws)
        join_state = get_call_expr_arg(
            "join_probe_consume_batch", args, kws, 0, "join_state"
        )
        StreamingStateType.ensure_known_inputs(
            "join_probe_consume_batch",
            (join_state.build_table_type, join_state.probe_table_type),
        )
        out_table_type = join_state.output_type
        # Output is (out_table, out_is_last, request_input)
        output_type = types.BaseTuple.from_types(
            (out_table_type, types.bool_, types.bool_)
        )

        pysig = numba.core.utils.pysignature(join_probe_consume_batch)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(output_type, *folded_args).replace(pysig=pysig)


JoinProbeConsumeBatchInfer._no_unliteral = True


@lower_builtin(join_probe_consume_batch, types.VarArg(types.Any))
def lower_join_probe_consume_batch(context, builder, sig, args):
    """lower join_probe_consume_batch() using gen_join_probe_consume_batch_impl above"""
    impl = gen_join_probe_consume_batch_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


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


@intrinsic
def get_op_pool_budget_bytes(
    typingctx,
    join_state,
):
    """
    Get the current budget (in bytes) of this join operator.
    This is only used for testing and debugging purposes.
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(64),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="join_get_op_pool_budget_bytes"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.uint64(join_state)
    return sig, codegen


@intrinsic
def get_op_pool_bytes_pinned(
    typingctx,
    join_state,
):
    """
    Get the number of bytes currently pinned by the
    OperatorBufferPool of this join operator.
    This is only used for testing and debugging purposes.
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(64),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="join_get_op_pool_bytes_pinned"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.uint64(join_state)
    return sig, codegen


@intrinsic
def get_op_pool_bytes_allocated(
    typingctx,
    join_state,
):
    """
    Get the number of bytes currently allocated by the
    OperatorBufferPool of this join operator.
    This is only used for testing and debugging purposes.
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(64),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="join_get_op_pool_bytes_allocated"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.uint64(join_state)
    return sig, codegen


@intrinsic
def get_num_partitions(
    typingctx,
    join_state,
):
    """
    Get the number of partitions of this join operator.
    This is only used for testing and debugging purposes.
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(32),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="join_get_num_partitions"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.uint32(join_state)
    return sig, codegen


@intrinsic
def get_partition_num_top_bits_by_idx(typingctx, join_state, idx):
    """
    Get the number of bits in the 'top_bitmask' of a partition of this join
    operator by the partition index.
    This is only used for testing and debugging purposes.
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(32),
            [lir.IntType(8).as_pointer(), lir.IntType(64)],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="join_get_partition_num_top_bits_by_idx"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.uint32(join_state, idx)
    return sig, codegen


@intrinsic
def get_partition_top_bitmask_by_idx(typingctx, join_state, idx):
    """
    Get the 'top_bitmask' of a partition of this join operator by the partition index.
    This is only used for testing and debugging purposes.
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(32),
            [lir.IntType(8).as_pointer(), lir.IntType(64)],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="join_get_partition_top_bitmask_by_idx"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.uint32(join_state, idx)
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_partition_state(join_state):
    """
    Get the partition state (the number of bits in the 'top_bitmask' and 'top_bitmask')
    of all partitions of this join operator.
    This is only used for testing and debugging purposes.
    """

    def impl(join_state):  # pragma: no cover
        partition_state = []
        for idx in range(get_num_partitions(join_state)):
            partition_state.append(
                (
                    get_partition_num_top_bits_by_idx(join_state, idx),
                    get_partition_top_bitmask_by_idx(join_state, idx),
                )
            )
        return partition_state

    return impl
