# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Support for streaming groupby (a.k.a. vectorized groupby).
This file is mostly wrappers for C++ implementations.
"""
from functools import cached_property
from typing import Dict, List, Set, Tuple

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
from bodo.ext import stream_groupby_cpp
from bodo.ir.aggregate import supported_agg_funcs
from bodo.libs.array import (
    cpp_table_to_py_table,
    delete_table,
    py_data_to_cpp_table,
)
from bodo.libs.array import table_type as cpp_table_type
from bodo.libs.stream_base import StreamingStateType
from bodo.utils.transform import get_call_expr_arg
from bodo.utils.typing import (
    BodoError,
    MetaType,
    error_on_unsupported_streaming_arrays,
    is_overload_none,
    to_nullable_type,
    unwrap_typeref,
)

ll.add_symbol(
    "groupby_state_init_py_entry", stream_groupby_cpp.groupby_state_init_py_entry
)
ll.add_symbol(
    "grouping_sets_state_init_py_entry",
    stream_groupby_cpp.grouping_sets_state_init_py_entry,
)
ll.add_symbol(
    "groupby_build_consume_batch_py_entry",
    stream_groupby_cpp.groupby_build_consume_batch_py_entry,
)
ll.add_symbol(
    "grouping_sets_build_consume_batch_py_entry",
    stream_groupby_cpp.grouping_sets_build_consume_batch_py_entry,
)
ll.add_symbol(
    "groupby_produce_output_batch_py_entry",
    stream_groupby_cpp.groupby_produce_output_batch_py_entry,
)
ll.add_symbol(
    "grouping_sets_produce_output_batch_py_entry",
    stream_groupby_cpp.grouping_sets_produce_output_batch_py_entry,
)
ll.add_symbol("delete_groupby_state", stream_groupby_cpp.delete_groupby_state)
ll.add_symbol(
    "delete_grouping_sets_state", stream_groupby_cpp.delete_grouping_sets_state
)

ll.add_symbol(
    "end_union_consume_pipeline_py_entry",
    stream_groupby_cpp.end_union_consume_pipeline_py_entry,
)

# The following are used for debugging and testing purposes only:
ll.add_symbol(
    "groupby_get_op_pool_bytes_pinned", stream_groupby_cpp.get_op_pool_bytes_pinned
)
ll.add_symbol(
    "groupby_get_op_pool_bytes_allocated",
    stream_groupby_cpp.get_op_pool_bytes_allocated,
)
ll.add_symbol(
    "groupby_get_num_partitions",
    stream_groupby_cpp.get_num_partitions,
)
ll.add_symbol(
    "groupby_get_partition_num_top_bits_by_idx",
    stream_groupby_cpp.get_partition_num_top_bits_by_idx,
)
ll.add_symbol(
    "groupby_get_partition_top_bitmask_by_idx",
    stream_groupby_cpp.get_partition_top_bitmask_by_idx,
)


class GroupbyStateType(StreamingStateType):
    """Type for C++ GroupbyState pointer"""

    def __init__(
        self,
        key_inds,
        grouping_sets,
        fnames,
        f_in_offsets,
        f_in_cols,
        mrnf_sort_col_inds: tuple[int] = (),
        mrnf_sort_col_asc: tuple[int] = (),
        mrnf_sort_col_na: tuple[int] = (),
        mrnf_col_inds_keep: tuple[int] = (),
        build_table_type=types.unknown,
    ):
        error_on_unsupported_streaming_arrays(build_table_type)

        self.key_inds = key_inds
        self.grouping_sets = grouping_sets
        self.fnames = fnames
        self.f_in_offsets = f_in_offsets
        self.f_in_cols = f_in_cols
        self.mrnf_sort_col_inds: tuple[int] = mrnf_sort_col_inds
        self.mrnf_sort_col_asc: tuple[int] = mrnf_sort_col_asc
        self.mrnf_sort_col_na: tuple[int] = mrnf_sort_col_na
        self.mrnf_col_inds_keep: tuple[int] = mrnf_col_inds_keep
        self.build_table_type = build_table_type
        self.uses_grouping_sets = len(grouping_sets) > 1 or grouping_sets[0] != key_inds
        super().__init__(
            f"GroupbyStateType({key_inds=}, {grouping_sets=}, {fnames=}, {f_in_offsets=}, "
            f"{f_in_cols=}, {mrnf_sort_col_inds=}, {mrnf_sort_col_asc=}, {mrnf_sort_col_na=}, "
            f"{mrnf_col_inds_keep=}, build_table={build_table_type})"
        )

    @property
    def key(self):
        return (
            self.key_inds,
            self.grouping_sets,
            self.fnames,
            self.f_in_offsets,
            self.f_in_cols,
            self.build_table_type,
            self.mrnf_sort_col_inds,
            self.mrnf_sort_col_asc,
            self.mrnf_sort_col_na,
            self.mrnf_col_inds_keep,
        )

    @property
    def n_keys(self):
        return len(self.key_inds)

    @property
    def n_mrnf_sort_keys(self):
        return len(self.mrnf_sort_col_inds)

    @property
    def ftypes(self):
        func_types = []
        for fname in self.fnames:
            if fname not in supported_agg_funcs or (
                not self.uses_grouping_sets and fname == "grouping"
            ):
                # Note: While 'grouping' could be supported in regular group by, it really only makes
                # sense in the context of grouping sets (and is only supported there). In general it should
                # be simplified to a constant value in the non-grouping sets case.
                raise BodoError(fname + " is not a supported aggregate function.")
            func_types.append(supported_agg_funcs.index(fname))
        return func_types

    @cached_property
    def _col_reorder_map(self) -> Dict[int, int]:
        """
        Generate a mapping to the input components from
        the Python types to the runtime C++ input type.

        Returns:
            Dict[int, int]: A dictionary containing the column remapping.
        """
        return {idx: i for i, idx in enumerate(self.build_indices)}

    @property
    def reordered_f_in_cols(self) -> Tuple[int]:
        """
        Because we reorder the columns to put the keys in the front, we need to
        map the original column indices contained in f_in_cols to the new column
        indices after reordering.

        In the case that the build_table_type hasn't been resolved yet, we just
        return the original f_in_cols.

        Returns:
            Tuple[int]: A tuple with the f_in_cols after remapping
        """
        if self.build_table_type == types.unknown:
            return self.f_in_cols
        return tuple([self._col_reorder_map[i] for i in self.f_in_cols])

    @cached_property
    def force_nullable_keys(self) -> Set[int]:
        """Determine the set of key indices that may be
        missing from a grouping set and therefore must be
        converted to a nullable type.

        Returns:
            Set[int]: The set of keys that must be nullable because
            they are missing from at least 1 grouping set.
        """
        nullable_keys = set()
        keys_set = set(self.key_inds)
        for grouping_set in self.grouping_sets:
            nullable_keys.update(keys_set - set(grouping_set))
        return nullable_keys

    @cached_property
    def key_types(self) -> List[types.ArrayCompatible]:
        """Generate the list of array types that should be used for the
        keys to groupby.

        Returns:
            List[types.ArrayCompatible]: The list of array types used
            by the keys.
        """
        build_table_type = self.build_table_type
        if build_table_type == types.unknown:
            # Typing transformations haven't fully finished yet.
            return []

        build_key_inds = self.key_inds
        arr_types = []
        num_keys = len(build_key_inds)
        nullable_keys = self.force_nullable_keys

        for i in range(num_keys):
            build_key_index = build_key_inds[i]
            build_arr_type = self.build_table_type.arr_types[build_key_index]
            if build_key_index in nullable_keys:
                build_arr_type = to_nullable_type(build_arr_type)
            arr_types.append(build_arr_type)

        return arr_types

    @property
    def key_casted_table_type(self) -> bodo.hiframes.table.TableType:
        """Returns the type of a table after casting any keys to nullable
        due to missing keys from grouping sets.

        Returns:
            bodo.hiframes.table.TableType: The output table type.
        """
        build_table_type = self.build_table_type
        if build_table_type == types.unknown:
            # Typing transformations haven't fully finished yet.
            return build_table_type

        total_arr_types = []
        nullable_keys = self.force_nullable_keys
        for i, arr_type in enumerate(build_table_type.arr_types):
            if i in nullable_keys:
                arr_type = to_nullable_type(arr_type)
            total_arr_types.append(arr_type)
        return bodo.hiframes.table.TableType(tuple(total_arr_types))

    @cached_property
    def mrnf_sort_col_types(self) -> list[types.ArrayCompatible]:
        """
        Generate the list of array types that should be used for
        the order-by columns in the MRNF case.
        This is similar to the 'key_types' function above (except
        it's for the order-by columns in MRNF) and
        follows the same assumptions.
        Returns an empty list in the non-MRNF case.

        Returns:
            list[types.ArrayCompatible]: The list of array types
            used by the order-by columns.
        """
        build_table_type = self.build_table_type
        if build_table_type == types.unknown:
            # Typing transformations haven't fully finished yet.
            return []
        mrnf_sort_col_inds = self.mrnf_sort_col_inds

        num_sort_cols = len(mrnf_sort_col_inds)
        arr_types = [
            self.build_table_type.arr_types[mrnf_sort_col_inds[i]]
            for i in range(num_sort_cols)
        ]

        return arr_types

    @staticmethod
    def _derive_input_type(
        key_types: list[types.ArrayCompatible],
        key_indices: tuple[int],
        mrnf_sort_col_types: list[types.ArrayCompatible],
        mrnf_sort_col_indices: tuple[int],
        table_type: bodo.hiframes.table.TableType,
    ) -> List[types.ArrayCompatible]:
        """Generate the input table type based on the given key types, key
        indices, and table type.

        Args:
            key_types (List[types.ArrayCompatible]): The list of key types in order.
            key_indices (N Tuple(int)): The indices of the key columns. These are the partition
                columns in the MRNF case.
            mrnf_sort_col_types (List[types.ArrayCompatible]): The list of MRNF sort column
                types in the MRNF case.
            mrnf_sort_col_indices (N Tuple(int)): The indices of the sort columns in the MRNF case.
            table_type (TableType): The input table type.

        Returns:
            List[types.ArrayCompatible]: The list of array types for the input C++ table (in order).
        """

        # The columns are: [<key/mrnf-partition columns>, <mrnf-orderby columns>, <rest of the columns>]
        types = key_types.copy()
        types.extend(mrnf_sort_col_types)
        idx_set = set(list(key_indices) + list(mrnf_sort_col_indices))

        # Append the data columns
        for i in range(len(table_type.arr_types)):
            if i not in idx_set:
                types.append(table_type.arr_types[i])
        return types

    @cached_property
    def build_reordered_arr_types(self) -> List[types.ArrayCompatible]:
        """
        Get the list of array types for the actual input to the C++ build table.
        This is different from the build_table_type because the input to the C++
        will reorder keys to the front. In the MRNF case, it will put the
        partition and sort columns in the front (in that order).
        The sort columns will be in the required sort order.

        Returns:
            List[types.ArrayCompatible]: The list of array types for the build table.
        """
        if self.build_table_type == types.unknown:
            return []

        key_types = self.key_types
        key_indices = self.key_inds
        mrnf_sort_col_types = self.mrnf_sort_col_types
        mrnf_sort_col_indices = self.mrnf_sort_col_inds
        table = self.key_casted_table_type
        return self._derive_input_type(
            key_types, key_indices, mrnf_sort_col_types, mrnf_sort_col_indices, table
        )

    @property
    def build_arr_ctypes(self) -> np.ndarray:
        """
        Fetch the CTypes used for each array in the build table.

        Note: We must use build_reordered_arr_types to account for reordering.

        Returns:
            List(int): The ctypes for each array in the build table. Note
                that C++ wants the actual integer but these are the values derived from
                CTypeEnum.
        """
        return self._derive_c_types(self.build_reordered_arr_types)

    @property
    def build_arr_array_types(self) -> np.ndarray:
        """
        Fetch the CArrayTypeEnum used for each array in the build table.

        Note: We must use build_reordered_arr_types to account for reordering.


        Returns:
            List(int): The CArrayTypeEnum for each array in the build table. Note
                that C++ wants the actual integer but these are the values derived from
                CArrayTypeEnum.
        """
        return self._derive_c_array_types(self.build_reordered_arr_types)

    @staticmethod
    def _derive_cpp_indices(key_indices, mrnf_sort_col_inds, num_cols):
        """Generate the indices used for the C++ table from the
        given Python table.

        Args:
            key_indices (N Tuple(int)): The indices of the key columns.
            mrnf_sort_col_inds (tuple[int]): The indices of the order-by
                columns for MRNF (if this is the MRNF case). In the non-MRNF
                case, this must be an empty tuple.
            num_cols (int): The number of total columns in the table.

        Returns:
            N Tuple(int): Tuple giving the order of the output indices
        """
        total_idxs = []
        for key_idx in key_indices:
            total_idxs.append(key_idx)

        for mrnf_sort_idx in mrnf_sort_col_inds:
            total_idxs.append(mrnf_sort_idx)

        idx_set = set(list(key_indices) + list(mrnf_sort_col_inds))
        for i in range(num_cols):
            if i not in idx_set:
                total_idxs.append(i)
        return tuple(total_idxs)

    @property
    def mrnf_cols_to_keep_bitmask(self) -> List[bool]:
        """Creates a bitmask for MRNF key and sort columns.

        Returns:
            A bitmask of length n_sort_keys + n_keys where True indicates that we should keep that column in the output.
        """
        n_build_arrs = len(self.build_arr_array_types)
        cols_to_keep_bitmask = [idx in self.mrnf_col_inds_keep for idx in self.key_inds]
        cols_to_keep_bitmask.extend(
            idx in self.mrnf_col_inds_keep for idx in self.mrnf_sort_col_inds
        )
        cols_to_keep_bitmask.extend(
            [True for _ in range(n_build_arrs - len(cols_to_keep_bitmask))]
        )
        return cols_to_keep_bitmask

    @cached_property
    def build_indices(self):
        if self.build_table_type == types.unknown:
            return ()

        return self._derive_cpp_indices(
            self.key_inds, self.mrnf_sort_col_inds, len(self.build_table_type.arr_types)
        )

    @cached_property
    def out_table_type(self):
        if self.build_table_type == types.unknown:
            return types.unknown

        # TODO[BSE-578]: get proper output type for all functions
        out_arr_types = []
        if self.fnames != ("min_row_number_filter",):
            for i, f_name in enumerate(self.fnames):
                if f_name != "grouping":
                    assert (
                        self.f_in_offsets[i + 1] == self.f_in_offsets[i] + 1
                    ), "only functions with single input column expect grouping supported in streaming groupby currently"
                # Note: Use f_in_cols because we need the original column location before reordering
                # for C++.
                in_type = self.build_table_type.arr_types[
                    self.f_in_cols[self.f_in_offsets[i]]
                ]
                (
                    out_type,
                    err_msg,
                ) = bodo.hiframes.pd_groupby_ext.get_groupby_output_dtype(
                    in_type, f_name
                )
                assert err_msg == "ok", "Function typing failed in streaming groupby"
                out_arr_types.append(out_type)
            return bodo.TableType(tuple(self.key_types + out_arr_types))
        else:
            # In the MRNF case, it will simply be all the indices in
            # 'self.mrnf_col_inds_keep'.
            # Note that this is the python table order. The C++
            # output will be in a different order:
            # [<partition_cols_to_keep>, <sort_cols_to_keep>, <remaining_cols>]
            # and will be re-arranged accordingly. See
            # 'cpp_output_table_to_py_table_idx_map' for how this reordering
            # is computed.

            for i in range(len(self.build_table_type.arr_types)):
                if i in self.mrnf_col_inds_keep:
                    out_arr_types.append(self.build_table_type.arr_types[i])
            return bodo.TableType(tuple(out_arr_types))

    @cached_property
    def cpp_output_table_to_py_table_idx_map(self) -> list[int]:
        """
        Ordered list of indices into the C++ output table
        to use for constructing the Python output table.
        This is essentially a NOP in the non-MRNF case since
        the order of the columns in the C++ and Python table
        is the same. In the MRNF, the C++ output is in the order:
        [<partition_cols_to_keep>, <sort_cols_to_keep>, <remaining_cols>]
        and we must re-order it to match the input table
        (i.e. 'mrnf_col_inds_keep') since MRNF is supposed to act
        as a 'filter'.

        Returns:
            list[int]: Ordered list of indices.
        """
        out_table_type = self.out_table_type
        # TODO[BSE-645]: Support pruning output columns.
        num_cols = len(out_table_type.arr_types)

        if self.fnames != ("min_row_number_filter",):
            # In the non-MRNF case, the mapping is trivial,
            # i.e. no re-ordering of columns is required.
            num_cols = len(out_table_type.arr_types)
            return list(range(num_cols))
        else:
            # In the MRNF case, we receive the output in the
            # order:
            # [<part_cols_to_keep>, <sort_cols_to_keep>, <remaining_cols>].
            # We need to create a reverse map from this
            # to self.mrnf_col_inds_keep.

            # First construct what the cpp output order would be:
            cpp_out_order = []
            seen_indices = set()
            for idx in self.key_inds:
                if idx in self.mrnf_col_inds_keep:
                    cpp_out_order.append(idx)
                    seen_indices.add(idx)
            for idx in self.mrnf_sort_col_inds:
                if idx in self.mrnf_col_inds_keep:
                    cpp_out_order.append(idx)
                    seen_indices.add(idx)
            for idx in self.mrnf_col_inds_keep:
                if idx not in seen_indices:
                    cpp_out_order.append(idx)

            # Now map the C++ order to the desired order:
            cpp_table_to_py_map = [
                cpp_out_order.index(idx) for idx in self.mrnf_col_inds_keep
            ]
            return cpp_table_to_py_map


register_model(GroupbyStateType)(models.OpaqueModel)


@intrinsic
def _init_groupby_state(
    typingctx,
    operator_id,
    build_arr_dtypes,
    build_arr_array_types,
    n_build_arrs,
    ftypes_t,
    window_ftypes_t,
    f_in_offsets_t,
    f_in_cols_t,
    n_funcs_t,
    mrnf_sort_asc_t,
    mrnf_sort_na_t,
    mrnf_n_sort_keys_t,
    cols_to_keep_t,
    window_args,
    op_pool_size_bytes_t,
    output_state_type,
    parallel_t,
):
    """Initialize C++ GroupbyState pointer

    Args:
        operator_id (int64): ID of this operator (used for looking up budget),
        build_arr_dtypes (int8*): pointer to array of ints representing array dtypes
                                   (as provided by numba_to_c_type)
        build_arr_array_types (int8*): pointer to array of ints representing array types
                                    (as provided by numba_to_c_array_type)
        n_build_arrs (int32): number of build columns
        op_pool_size_bytes_t (int64): Number of pinned bytes that this operator is allowed
             to use. Set this to -1 to let the operator use a pre-determined portion of
             the total available memory.
        ftypes (int32*): List of aggregate functions to use
        f_in_offsets (int32*): Offsets into f_in_cols for the aggregate functions.
        f_in_cols (int32*): Columns for the aggregate functions.
        n_funcs (int): Number of aggregate functions.
        mrnf_sort_asc (bool*): Bitmask for sort direction of order-by columns in MRNF case.
        mrnf_sort_na (bool*): Bitmask for null sort direction of order-by columns in MRNF case.
        mrnf_n_sort_keys (int): Number of MRNF order-by columns.
        cols_to_keep (bool*): Bitmask of columns to retain in output
            in the MRNF or window case.
        window_args (table_info*) table consisting of 1 row and a column for each scalar window arg
        op_pool_size_bytes (int64): Size of the operator pool (in bytes).
        output_state_type (TypeRef[GroupbyStateType]): The output type for the state
                                                    that should be generated.
    """
    output_type = unwrap_typeref(output_state_type)

    def codegen(context, builder, sig, args):
        (
            operator_id,
            build_arr_dtypes,
            build_arr_array_types,
            n_build_arrs,
            ftypes,
            window_ftypes,
            f_in_offsets,
            f_in_cols,
            n_funcs,
            mrnf_sort_asc,
            mrnf_sort_na,
            mrnf_n_sort_keys,
            cols_to_keep,
            window_args,
            op_pool_size_bytes,
            _,  # output_state_type
            parallel,
        ) = args
        n_keys = context.get_constant(types.uint64, output_type.n_keys)
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
                lir.IntType(32).as_pointer(),
                lir.IntType(32).as_pointer(),
                lir.IntType(32).as_pointer(),
                lir.IntType(32).as_pointer(),
                lir.IntType(32),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="groupby_state_init_py_entry"
        )
        input_args = (
            operator_id,
            build_arr_dtypes,
            build_arr_array_types,
            n_build_arrs,
            ftypes,
            window_ftypes,
            f_in_offsets,
            f_in_cols,
            n_funcs,
            n_keys,
            mrnf_sort_asc,
            mrnf_sort_na,
            mrnf_n_sort_keys,
            cols_to_keep,
            window_args,
            output_batch_size,
            parallel,
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
        types.CPointer(types.int32),
        types.CPointer(types.int32),
        types.CPointer(types.int32),
        types.CPointer(types.int32),
        types.int32,
        types.CPointer(types.bool_),
        types.CPointer(types.bool_),
        types.int64,
        types.CPointer(types.bool_),
        window_args,
        types.int64,
        output_state_type,
        parallel_t,
    )
    return sig, codegen


def init_groupby_state(
    operator_id,
    key_inds,
    fnames,  # fnames matches function names in supported_agg_funcs
    f_in_offsets,
    f_in_cols,
    mrnf_sort_col_inds=None,
    mrnf_sort_col_asc=None,
    mrnf_sort_col_na=None,
    mrnf_col_inds_keep=None,
    op_pool_size_bytes=-1,
    expected_state_type=None,
    parallel=False,
):
    pass


def _get_init_groupby_state_type(
    expected_state_type,
    key_inds,
    fnames,
    f_in_offsets,
    f_in_cols,
    mrnf_sort_col_inds,
    mrnf_sort_col_asc,
    mrnf_sort_col_na,
    mrnf_col_inds_keep,
):
    """Helper for init_groupby_state output typing that returns state type with unknown
    table types when expected state type is not provided.
    Also performs validation checks for MRNF and some other cases.
    """
    if is_overload_none(expected_state_type):
        key_inds = unwrap_typeref(key_inds).meta
        fnames = unwrap_typeref(fnames).meta
        f_in_offsets = unwrap_typeref(f_in_offsets).meta
        f_in_cols = unwrap_typeref(f_in_cols).meta
        # Check if the MRNF fields are provided:
        if not is_overload_none(mrnf_sort_col_inds):
            mrnf_sort_col_inds = unwrap_typeref(mrnf_sort_col_inds).meta
        else:
            mrnf_sort_col_inds = ()
        if not is_overload_none(mrnf_sort_col_asc):
            mrnf_sort_col_asc = unwrap_typeref(mrnf_sort_col_asc).meta
        else:
            mrnf_sort_col_asc = ()
        if not is_overload_none(mrnf_sort_col_na):
            mrnf_sort_col_na = unwrap_typeref(mrnf_sort_col_na).meta
        else:
            mrnf_sort_col_na = ()
        if not is_overload_none(mrnf_col_inds_keep):
            mrnf_col_inds_keep = unwrap_typeref(mrnf_col_inds_keep).meta
        else:
            mrnf_col_inds_keep = ()

        if len(mrnf_sort_col_inds) > 0:
            # In the MRNF case, if there are indices in both the
            # partition and sort columns, then raise an error.
            mrnf_sort_col_inds_set = set(list(mrnf_sort_col_inds))
            key_inds_set = set(list(key_inds))
            common_inds = mrnf_sort_col_inds_set.intersection(key_inds_set)
            if len(common_inds) > 0:
                raise BodoError(
                    "Groupby (Min Row-Number Filter): A column cannot be both a partition column and a sort column. "
                    f"The following column indices were in both sets: {common_inds}."
                )

        output_type = GroupbyStateType(
            key_inds,
            # A regular groupby has a single grouping set that matches the keys.
            (key_inds,),
            fnames,
            f_in_offsets,
            f_in_cols,
            mrnf_sort_col_inds,
            mrnf_sort_col_asc,
            mrnf_sort_col_na,
            mrnf_col_inds_keep,
        )
    else:
        output_type = expected_state_type

    ## Validation checks for the MRNF case (assuming typing transformations are done):
    if (
        (output_type.build_table_type != types.unknown)
        and len(output_type.fnames) > 0
        and output_type.fnames[0] == "min_row_number_filter"
    ):
        n_cols = len(output_type.build_indices)
        if len(output_type.fnames) > 1:
            raise BodoError(
                "Streaming Groupby: Min Row-Number Filter cannot be combined with other aggregation functions."
            )
        if len(output_type.f_in_cols) != (n_cols - output_type.n_keys):
            raise BodoError(
                "Groupby (Min Row-Number Filter): All columns except the partition columns must be in f_in_cols!"
            )
        expected_f_in_offsets = [0, len(output_type.f_in_cols)]
        if list(output_type.f_in_offsets) != expected_f_in_offsets:
            raise BodoError(
                f"Groupby (Min Row-Number Filter): Expected f_in_offsets to be '{expected_f_in_offsets}', "
                f"but got '{list(output_type.f_in_offsets)}' instead"
            )
        if not (
            output_type.n_mrnf_sort_keys
            == len(output_type.mrnf_sort_col_asc)
            == len(output_type.mrnf_sort_col_na)
        ):
            raise BodoError(
                "Groupby (Min Row-Number Filter): Mismatch in expected sizes of arguments! "
                f"n_mrnf_sort_keys: {output_type.n_mrnf_sort_keys}, "
                f"len(mrnf_sort_col_asc): {len(output_type.mrnf_sort_col_asc)}, "
                f"len(mrnf_sort_col_na): {len(output_type.mrnf_sort_col_na)}."
            )

        # If an index is neither a partition column nor a sort column, then it must
        # be in the list of indices to keep. Otherwise that index is not required and
        # should never have been passed in the first place.
        partition_or_sort_col_set = set(
            list(output_type.key_inds) + list(output_type.mrnf_sort_col_inds)
        )
        mrnf_col_inds_keep_set = set(list(output_type.mrnf_col_inds_keep))
        for idx in range(n_cols):
            if (idx not in partition_or_sort_col_set) and (
                idx not in mrnf_col_inds_keep_set
            ):
                raise BodoError(
                    f"Groupby (Min Row-Number Filter): Column {idx} must be in the list of indices to "
                    "keep since it's neither a partition columns nor a sort column!"
                )

    for i in range(len(output_type.fnames)):
        if output_type.build_table_type == types.unknown:
            # Typing transformations haven't fully finished yet.
            break

        if output_type.fnames[i] == "min_row_number_filter":
            # In the MRNF case, the sort columns cannot be semi-structured. The rest of the columns
            # can be.
            # (Ticket for adding this support: https://bodo.atlassian.net/browse/BSE-2599)
            for sort_col_idx in output_type.mrnf_sort_col_inds:
                col_arr_type = output_type.build_table_type.arr_types[sort_col_idx]
                if isinstance(
                    col_arr_type,
                    (bodo.MapArrayType, bodo.ArrayItemArrayType, bodo.StructArrayType),
                ):
                    raise BodoError(
                        "Groupby (Min Row-Number Filter): Sorting on semi-structured arrays is not supported."
                    )

        else:
            # If there are any semi-structured arrays, we only support first, count and size:
            supported_nested_agg_funcs = ["first", "count", "size"]
            for idx in output_type.f_in_offsets[i : i + 1]:
                # Note: Use f_in_cols because we need the original column location before reordering
                # for C++.
                col_arr_type = output_type.build_table_type.arr_types[
                    output_type.f_in_cols[output_type.f_in_offsets[idx]]
                ]
                if (
                    isinstance(
                        col_arr_type,
                        (
                            bodo.MapArrayType,
                            bodo.ArrayItemArrayType,
                            bodo.StructArrayType,
                        ),
                    )
                    and output_type.fnames[i] not in supported_nested_agg_funcs
                ):
                    raise BodoError(
                        f"Groupby does not support semi-structured arrays for aggregations other than {', '.join(supported_nested_agg_funcs[:-1])} and {supported_nested_agg_funcs[-1]}."
                    )

    return output_type


@infer_global(init_groupby_state)
class InitGroupbyStateInfer(AbstractTemplate):
    """Typer for init_groupby_state that returns state type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(init_groupby_state)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        expected_state_type = unwrap_typeref(folded_args[10])
        (
            key_inds,
            fnames,
            f_in_offsets,
            f_in_cols,
            mrnf_sort_col_inds,
            mrnf_sort_col_asc,
            mrnf_sort_col_na,
            mrnf_col_inds_keep,
        ) = folded_args[1:9]
        output_type = _get_init_groupby_state_type(
            expected_state_type,
            key_inds,
            fnames,
            f_in_offsets,
            f_in_cols,
            mrnf_sort_col_inds,
            mrnf_sort_col_asc,
            mrnf_sort_col_na,
            mrnf_col_inds_keep,
        )
        return signature(output_type, *folded_args).replace(pysig=pysig)


InitGroupbyStateInfer._no_unliteral = True


@lower_builtin(init_groupby_state, types.VarArg(types.Any))
def lower_init_groupby_state(context, builder, sig, args):
    """lower init_groupby_state() using gen_init_groupby_state_impl above"""
    impl = gen_init_groupby_state_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def gen_init_groupby_state_impl(
    operator_id,
    key_inds,
    fnames,  # fnames matches function names in supported_agg_funcs
    f_in_offsets,
    f_in_cols,
    mrnf_sort_col_inds=None,
    mrnf_sort_col_asc=None,
    mrnf_sort_col_na=None,
    mrnf_col_inds_keep=None,
    op_pool_size_bytes=-1,
    expected_state_type=None,
    parallel=False,
):
    expected_state_type = unwrap_typeref(expected_state_type)

    output_type = _get_init_groupby_state_type(
        expected_state_type,
        key_inds,
        fnames,
        f_in_offsets,
        f_in_cols,
        mrnf_sort_col_inds,
        mrnf_sort_col_asc,
        mrnf_sort_col_na,
        mrnf_col_inds_keep,
    )

    build_arr_dtypes = output_type.build_arr_ctypes
    build_arr_array_types = output_type.build_arr_array_types
    n_build_arrs = len(build_arr_array_types)
    ftypes = output_type.ftypes

    ftypes_arr = np.array(ftypes, np.int32)
    window_ftypes_arr = np.array([], np.int32)
    f_in_offsets_arr = np.array(output_type.f_in_offsets, np.int32)
    f_in_cols_arr = np.array(output_type.reordered_f_in_cols, np.int32)
    n_funcs = len(output_type.fnames)

    mrnf_sort_asc_arr = np.array(output_type.mrnf_sort_col_asc, dtype=np.bool_)
    mrnf_sort_na_arr = np.array(output_type.mrnf_sort_col_na, dtype=np.bool_)

    mrnf_cols_to_keep_arr = np.array(output_type.mrnf_cols_to_keep_bitmask, np.bool_)

    mrnf_n_sort_keys = output_type.n_mrnf_sort_keys

    def impl_init_groupby_state(
        operator_id,
        key_inds,
        fnames,
        f_in_offsets,
        f_in_cols,
        mrnf_sort_col_inds=None,
        mrnf_sort_col_asc=None,
        mrnf_sort_col_na=None,
        mrnf_col_inds_keep=None,
        op_pool_size_bytes=-1,
        expected_state_type=None,
        parallel=False,
    ):  # pragma: no cover
        return _init_groupby_state(
            operator_id,
            build_arr_dtypes.ctypes,
            build_arr_array_types.ctypes,
            n_build_arrs,
            ftypes_arr.ctypes,
            window_ftypes_arr.ctypes,
            f_in_offsets_arr.ctypes,
            f_in_cols_arr.ctypes,
            n_funcs,
            mrnf_sort_asc_arr.ctypes,
            mrnf_sort_na_arr.ctypes,
            mrnf_n_sort_keys,
            mrnf_cols_to_keep_arr.ctypes,
            None,  # window_args
            op_pool_size_bytes,
            output_type,
            parallel,
        )

    return impl_init_groupby_state


@intrinsic
def _init_grouping_sets_state(
    typingctx,
    operator_id,
    sub_operator_ids,
    build_arr_dtypes,
    build_arr_array_types,
    n_build_arrs,
    grouping_sets_data_arr_t,
    grouping_sets_offset_arr_t,
    n_grouping_sets,
    ftypes_t,
    f_in_offsets_t,
    f_in_cols_t,
    n_funcs_t,
    output_state_type,
    parallel_t,
):
    """Initialize C++ GroupingSetsState pointer

    Args:
        operator_id (int64): ID of this operator (used for looking up budget),
        sub_operator_ids (int64*): IDs of the sub-operators for each group by state
            (used for looking up budget). There is always 1 per grouping set.
        build_arr_dtypes (int8*): pointer to array of ints representing array dtypes
                                   (as provided by numba_to_c_type)
        build_arr_array_types (int8*): pointer to array of ints representing array types
                                    (as provided by numba_to_c_array_type)
        n_build_arrs (int32): number of build columns
        grouping_sets_data_arr_t (int32*): Flatten data array of the keys used by each
            grouping set.
        grouping_sets_offset_arr_t (int32*): Offset array into the data array for each
            grouping set. Grouping set i uses offsets i and i+1.
        n_grouping_sets (int64): Number of grouping sets.
        ftypes (int32*): List of aggregate functions to use
        f_in_offsets (int32*): Offsets into f_in_cols for the aggregate functions.
        f_in_cols (int32*): Columns for the aggregate functions.
        n_funcs (int32): Number of aggregate functions.
        output_state_type (TypeRef[GroupbyStateType]): The output type for the state
                                                    that should be generated.
        parallel_t (bool): Whether to run in parallel.
    """
    output_type = unwrap_typeref(output_state_type)

    def codegen(context, builder, sig, args):
        (
            operator_id,
            sub_operator_ids,
            build_arr_dtypes,
            build_arr_array_types,
            n_build_arrs,
            grouping_sets_data,
            grouping_sets_offsets,
            n_grouping_sets,
            ftypes,
            f_in_offsets,
            f_in_cols,
            n_funcs,
            _,  # output_state_type
            parallel,
        ) = args
        n_keys = context.get_constant(types.uint64, output_type.n_keys)
        output_batch_size = context.get_constant(
            types.int64, bodo.bodosql_streaming_batch_size
        )
        sync_iter = context.get_constant(types.int64, bodo.stream_loop_sync_iters)
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(64),
                lir.IntType(64).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(32).as_pointer(),
                lir.IntType(32).as_pointer(),
                lir.IntType(64),
                lir.IntType(32).as_pointer(),
                lir.IntType(32).as_pointer(),
                lir.IntType(32).as_pointer(),
                lir.IntType(32),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(64),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="grouping_sets_state_init_py_entry"
        )
        input_args = (
            operator_id,
            sub_operator_ids,
            build_arr_dtypes,
            build_arr_array_types,
            n_build_arrs,
            grouping_sets_data,
            grouping_sets_offsets,
            n_grouping_sets,
            ftypes,
            f_in_offsets,
            f_in_cols,
            n_funcs,
            n_keys,
            output_batch_size,
            parallel,
            sync_iter,
        )
        ret = builder.call(fn_tp, input_args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = output_type(
        types.int64,
        types.CPointer(types.int64),
        types.voidptr,
        types.voidptr,
        types.int32,
        types.CPointer(types.int32),
        types.CPointer(types.int32),
        types.int64,
        types.CPointer(types.int32),
        types.CPointer(types.int32),
        types.CPointer(types.int32),
        types.int32,
        output_state_type,
        parallel_t,
    )
    return sig, codegen


def init_grouping_sets_state(
    operator_id,
    sub_operator_ids,
    key_inds,
    grouping_sets,
    fnames,  # fnames matches function names in supported_agg_funcs
    f_in_offsets,
    f_in_cols,
    expected_state_type=None,
    parallel=False,
):
    pass


def _get_init_grouping_sets_output_type(
    expected_state_type, key_inds, grouping_sets, fnames, f_in_offsets, f_in_cols
):
    """Helper for init_grouping_sets_state output typing that returns state type with
    unknown table types when expected state type is not provided.
    """
    if is_overload_none(expected_state_type):
        key_inds = unwrap_typeref(key_inds).meta
        grouping_sets = unwrap_typeref(grouping_sets).meta
        fnames = unwrap_typeref(fnames).meta
        f_in_offsets = unwrap_typeref(f_in_offsets).meta
        f_in_cols = unwrap_typeref(f_in_cols).meta

        output_type = GroupbyStateType(
            key_inds,
            grouping_sets,
            fnames,
            f_in_offsets,
            f_in_cols,
        )
    else:
        output_type = expected_state_type

    return output_type


@infer_global(init_grouping_sets_state)
class InitGroupingSetsStateInfer(AbstractTemplate):
    """Typer for init_grouping_sets_state that returns state type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(init_grouping_sets_state)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        expected_state_type = unwrap_typeref(folded_args[7])
        (key_inds, grouping_sets, fnames, f_in_offsets, f_in_cols) = folded_args[2:7]
        output_type = _get_init_grouping_sets_output_type(
            expected_state_type,
            key_inds,
            grouping_sets,
            fnames,
            f_in_offsets,
            f_in_cols,
        )
        return signature(output_type, *folded_args).replace(pysig=pysig)


InitGroupingSetsStateInfer._no_unliteral = True


@lower_builtin(init_grouping_sets_state, types.VarArg(types.Any))
def lower_init_grouping_sets_state(context, builder, sig, args):
    """lower init_grouping_sets_state() using gen_init_grouping_sets_state_impl"""
    impl = gen_init_grouping_sets_state_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def gen_init_grouping_sets_state_impl(
    operator_id,
    sub_operator_ids,
    key_inds,
    grouping_sets,
    fnames,  # fnames matches function names in supported_agg_funcs
    f_in_offsets,
    f_in_cols,
    expected_state_type=None,
    parallel=False,
):
    expected_state_type = unwrap_typeref(expected_state_type)
    output_type = _get_init_grouping_sets_output_type(
        expected_state_type, key_inds, grouping_sets, fnames, f_in_offsets, f_in_cols
    )

    build_arr_dtypes = output_type.build_arr_ctypes
    build_arr_array_types = output_type.build_arr_array_types
    n_build_arrs = len(build_arr_array_types)
    ftypes = output_type.ftypes
    ftypes_arr = np.array(ftypes, np.int32)
    f_in_offsets_arr = np.array(output_type.f_in_offsets, np.int32)
    f_in_cols_arr = np.array(output_type.reordered_f_in_cols, np.int32)
    n_funcs = len(output_type.fnames)
    sub_operator_ids = unwrap_typeref(sub_operator_ids).meta
    sub_operator_id_arr = np.array(sub_operator_ids, np.int64)
    # Flatten the grouping sets into a single array for C++.
    # We pass this with 3 values:
    # 1. A data array containing the total grouping sets indices,
    # remapped for C++.
    # 2. An offsets array of length num_grouping_sets + 1
    # for indexing into the flattened array.
    # 3. The number of grouping sets.
    flatten_grouping_sets = []
    offsets = [0]
    for group in output_type.grouping_sets:
        offsets.append(offsets[-1] + len(group))
        # Remap the indices for C++. We use get because before typing is finished
        # the indices are not reordered yet.
        for index in group:
            flatten_grouping_sets.append(output_type._col_reorder_map.get(index, index))
    grouping_sets_arr = np.array(flatten_grouping_sets, np.int32)
    offsets_arr = np.array(offsets, np.int32)
    num_grouping_sets = len(output_type.grouping_sets)
    if num_grouping_sets <= 0:
        raise BodoError("Grouping sets must have at least one grouping set.")

    def impl_init_grouping_sets_state(
        operator_id,
        sub_operator_ids,
        key_inds,
        grouping_sets,
        fnames,  # fnames matches function names in supported_agg_funcs
        f_in_offsets,
        f_in_cols,
        expected_state_type=None,
        parallel=False,
    ):  # pragma: no cover
        return _init_grouping_sets_state(
            operator_id,
            sub_operator_id_arr.ctypes,
            build_arr_dtypes.ctypes,
            build_arr_array_types.ctypes,
            n_build_arrs,
            grouping_sets_arr.ctypes,
            offsets_arr.ctypes,
            num_grouping_sets,
            ftypes_arr.ctypes,
            f_in_offsets_arr.ctypes,
            f_in_cols_arr.ctypes,
            n_funcs,
            output_type,
            parallel,
        )

    return impl_init_grouping_sets_state


@intrinsic
def _groupby_build_consume_batch(
    typingctx,
    groupby_state,
    cpp_table,
    is_last,
    is_final_pipeline,
):
    def codegen(context, builder, sig, args):
        request_input = cgutils.alloca_once(builder, lir.IntType(1))
        fnty = lir.FunctionType(
            lir.IntType(1),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="groupby_build_consume_batch_py_entry"
        )
        ret = builder.call(fn_tp, tuple(args) + (request_input,))
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return context.make_tuple(
            builder, sig.return_type, [ret, builder.load(request_input)]
        )

    ret_type = types.Tuple([types.bool_, types.bool_])
    sig = ret_type(groupby_state, cpp_table, is_last, is_final_pipeline)
    return sig, codegen


def groupby_build_consume_batch(groupby_state, table, is_last, is_final_pipeline):
    pass


def gen_groupby_build_consume_batch_impl(
    groupby_state: GroupbyStateType, table, is_last, is_final_pipeline
):
    """Consume a build table batch in streaming groupby (insert into hash table and
    update running values)

    Args:
        groupby_state (GroupbyState): C++ GroupbyState pointer
        table (table_type): build table batch
        is_last (bool): is last batch (in this pipeline) locally
        is_final_pipeline (bool): Is this the final pipeline. Only relevant for the
         Union-Distinct case where this is called in multiple pipelines. For regular
         groupby, this should always be true.
    Returns:
        tuple(bool, bool): is last batch globally with possibility of false negatives
        due to iterations between syncs, whether to request input rows from preceding
         operators
    """
    in_col_inds = MetaType(groupby_state.build_indices)
    n_table_cols = len(in_col_inds)

    def impl_groupby_build_consume_batch(
        groupby_state, table, is_last, is_final_pipeline
    ):  # pragma: no cover
        cpp_table = py_data_to_cpp_table(table, (), in_col_inds, n_table_cols)
        return _groupby_build_consume_batch(
            groupby_state, cpp_table, is_last, is_final_pipeline
        )

    return impl_groupby_build_consume_batch


@infer_global(groupby_build_consume_batch)
class GroupbyBuildConsumeBatchInfer(AbstractTemplate):
    """Typer for groupby_build_consume_batch that returns bool as output type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(groupby_build_consume_batch)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        output_type = types.BaseTuple.from_types((types.bool_, types.bool_))
        return signature(output_type, *folded_args).replace(pysig=pysig)


@lower_builtin(groupby_build_consume_batch, types.VarArg(types.Any))
def lower_groupby_build_consume_batch(context, builder, sig, args):
    """lower groupby_build_consume_batch() using gen_groupby_build_consume_batch_impl above"""
    impl = gen_groupby_build_consume_batch_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


@intrinsic
def _groupby_grouping_sets_build_consume_batch(
    typingctx,
    groupby_state,
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
            builder.module, fnty, name="grouping_sets_build_consume_batch_py_entry"
        )
        ret = builder.call(fn_tp, tuple(args) + (request_input,))
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return context.make_tuple(
            builder, sig.return_type, [ret, builder.load(request_input)]
        )

    ret_type = types.Tuple([types.bool_, types.bool_])
    sig = ret_type(groupby_state, cpp_table, is_last)
    return sig, codegen


def groupby_grouping_sets_build_consume_batch(grouping_sets_state, table, is_last):
    pass


def gen_groupby_grouping_sets_build_consume_batch_impl(
    grouping_sets_state: GroupbyStateType, table, is_last
):
    """Consume a build table batch in streaming groupby (insert into hash table and
    update running values) with grouping sets.

    Args:
        grouping_sets_state (GroupbyState): C++ GroupingSetsState pointer
        table (table_type): build table batch
        is_last (bool): is last batch (in this pipeline) locally
    Returns:
        tuple(bool, bool): is last batch globally with possibility of false negatives
        due to iterations between syncs, whether to request input rows from preceding
         operators
    """
    in_col_inds = MetaType(grouping_sets_state.build_indices)
    n_table_cols = len(in_col_inds)

    cast_table_type = grouping_sets_state.key_casted_table_type

    def impl_groupby_build_consume_batch(
        grouping_sets_state, table, is_last
    ):  # pragma: no cover
        cast_table = bodo.utils.table_utils.table_astype(
            table, cast_table_type, False, False
        )
        cpp_table = py_data_to_cpp_table(cast_table, (), in_col_inds, n_table_cols)
        return _groupby_grouping_sets_build_consume_batch(
            grouping_sets_state, cpp_table, is_last
        )

    return impl_groupby_build_consume_batch


@infer_global(groupby_grouping_sets_build_consume_batch)
class GroupbyBuildConsumeGroupingSetsBatchInfer(AbstractTemplate):
    """Typer for groupby_grouping_sets_build_consume_batch that returns bool as output type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(groupby_grouping_sets_build_consume_batch)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        output_type = types.BaseTuple.from_types((types.bool_, types.bool_))
        return signature(output_type, *folded_args).replace(pysig=pysig)


@lower_builtin(groupby_grouping_sets_build_consume_batch, types.VarArg(types.Any))
def lower_groupby_grouping_sets_build_consume_batch(context, builder, sig, args):
    """lower groupby_grouping_sets_build_consume_batch() using gen_groupby_grouping_sets_build_consume_batch_impl above"""
    impl = gen_groupby_grouping_sets_build_consume_batch_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


@intrinsic
def _groupby_produce_output_batch(
    typingctx,
    groupby_state,
    produce_output,
):
    def codegen(context, builder, sig, args):
        out_is_last = cgutils.alloca_once(builder, lir.IntType(1))
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [lir.IntType(8).as_pointer(), lir.IntType(1).as_pointer(), lir.IntType(1)],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="groupby_produce_output_batch_py_entry"
        )
        func_args = [
            args[0],
            out_is_last,
            args[1],
        ]
        table_ret = builder.call(fn_tp, func_args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        items = [table_ret, builder.load(out_is_last)]
        return context.make_tuple(builder, sig.return_type, items)

    ret_type = types.Tuple([cpp_table_type, types.bool_])
    sig = ret_type(
        groupby_state,
        produce_output,
    )
    return sig, codegen


def groupby_produce_output_batch(groupby_state, produce_output):
    pass


def gen_groupby_produce_output_batch_impl(
    groupby_state: GroupbyStateType, produce_output
):
    """Produce output batches of groupby operation

    Args:
        groupby_state (GroupbyStateType): C++ GroupbyState pointer
        produce_output (bool): whether to produce output

    Returns:
        table_type: output table batch
        bool: global is last batch with possibility of false negatives due to iterations between syncs
    """
    out_table_type = groupby_state.out_table_type

    out_cols = groupby_state.cpp_output_table_to_py_table_idx_map
    out_cols_arr = np.array(out_cols, dtype=np.int64)

    def impl_groupby_produce_output_batch(
        groupby_state,
        produce_output,
    ):  # pragma: no cover
        out_cpp_table, out_is_last = _groupby_produce_output_batch(
            groupby_state, produce_output
        )
        out_table = cpp_table_to_py_table(
            out_cpp_table, out_cols_arr, out_table_type, 0
        )
        delete_table(out_cpp_table)
        return out_table, out_is_last

    return impl_groupby_produce_output_batch


@infer_global(groupby_produce_output_batch)
class GroupbyProduceOutputInfer(AbstractTemplate):
    """Typer for groupby_produce_output_batch that returns (output_table_type, bool)
    as output type.
    """

    def generic(self, args, kws):
        kws = dict(kws)
        groupby_state = get_call_expr_arg(
            "groupby_produce_output_batch", args, kws, 0, "groupby_state"
        )
        StreamingStateType.ensure_known_inputs(
            "groupby_produce_output_batch", (groupby_state.build_table_type,)
        )
        out_table_type = groupby_state.out_table_type
        # Output is (out_table, out_is_last)
        output_type = types.BaseTuple.from_types((out_table_type, types.bool_))

        pysig = numba.core.utils.pysignature(groupby_produce_output_batch)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(output_type, *folded_args).replace(pysig=pysig)


@lower_builtin(groupby_produce_output_batch, types.VarArg(types.Any))
def lower_groupby_produce_output_batch(context, builder, sig, args):
    """lower groupby_produce_output_batch() using gen_groupby_produce_output_batch_impl above"""
    impl = gen_groupby_produce_output_batch_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


@intrinsic
def _groupby_grouping_sets_produce_output_batch(
    typingctx,
    groupby_state,
    produce_output,
):
    def codegen(context, builder, sig, args):
        out_is_last = cgutils.alloca_once(builder, lir.IntType(1))
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [lir.IntType(8).as_pointer(), lir.IntType(1).as_pointer(), lir.IntType(1)],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="grouping_sets_produce_output_batch_py_entry"
        )
        func_args = [
            args[0],
            out_is_last,
            args[1],
        ]
        table_ret = builder.call(fn_tp, func_args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        items = [table_ret, builder.load(out_is_last)]
        return context.make_tuple(builder, sig.return_type, items)

    ret_type = types.Tuple([cpp_table_type, types.bool_])
    sig = ret_type(
        groupby_state,
        produce_output,
    )
    return sig, codegen


def groupby_grouping_sets_produce_output_batch(grouping_sets_state, produce_output):
    pass


def gen_groupby_grouping_sets_produce_output_batch_impl(
    grouping_sets_state: GroupbyStateType, produce_output
):
    """Produce output batches of groupby operation

    Args:
        grouping_sets_state (GroupbyStateType): C++ GroupingSetsState pointer
        produce_output (bool): whether to produce output

    Returns:
        table_type: output table batch
        bool: global is last batch with possibility of false negatives due to iterations between syncs
    """
    out_table_type = grouping_sets_state.out_table_type

    out_cols = grouping_sets_state.cpp_output_table_to_py_table_idx_map
    out_cols_arr = np.array(out_cols, dtype=np.int64)

    def impl_groupby_grouping_sets_produce_output_batch(
        grouping_sets_state,
        produce_output,
    ):  # pragma: no cover
        out_cpp_table, out_is_last = _groupby_grouping_sets_produce_output_batch(
            grouping_sets_state, produce_output
        )
        out_table = cpp_table_to_py_table(
            out_cpp_table, out_cols_arr, out_table_type, 0
        )
        delete_table(out_cpp_table)
        return out_table, out_is_last

    return impl_groupby_grouping_sets_produce_output_batch


@infer_global(groupby_grouping_sets_produce_output_batch)
class GroupbyProduceGroupingSetsOutputInfer(AbstractTemplate):
    """Typer for groupby_grouping_sets_produce_output_batch that returns (output_table_type, bool)
    as output type.
    """

    def generic(self, args, kws):
        kws = dict(kws)
        grouping_sets_state = get_call_expr_arg(
            "groupby_grouping_sets_produce_output_batch",
            args,
            kws,
            0,
            "grouping_sets_state",
        )
        StreamingStateType.ensure_known_inputs(
            "groupby_grouping_sets_produce_output_batch",
            (grouping_sets_state.build_table_type,),
        )
        out_table_type = grouping_sets_state.out_table_type
        # Output is (out_table, out_is_last)
        output_type = types.BaseTuple.from_types((out_table_type, types.bool_))

        pysig = numba.core.utils.pysignature(groupby_grouping_sets_produce_output_batch)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(output_type, *folded_args).replace(pysig=pysig)


@lower_builtin(groupby_grouping_sets_produce_output_batch, types.VarArg(types.Any))
def lower_groupby_grouping_sets_produce_output_batch(context, builder, sig, args):
    """lower groupby_grouping_sets_produce_output_batch() using gen_groupby_grouping_sets_produce_output_batch_impl above"""
    impl = gen_groupby_grouping_sets_produce_output_batch_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


@intrinsic
def delete_groupby_state(
    typingctx,
    groupby_state,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="delete_groupby_state"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    sig = types.void(groupby_state)
    return sig, codegen


@intrinsic
def delete_grouping_sets_state(
    typingctx,
    grouping_sets_state,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="delete_grouping_sets_state"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    sig = types.void(grouping_sets_state)
    return sig, codegen


@intrinsic
def get_op_pool_bytes_pinned(
    typingctx,
    groupby_state,
):
    """
    Get the number of bytes currently pinned by the
    OperatorBufferPool of this groupby operator.
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
            builder.module, fnty, name="groupby_get_op_pool_bytes_pinned"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.uint64(groupby_state)
    return sig, codegen


@intrinsic
def get_op_pool_bytes_allocated(
    typingctx,
    groupby_state,
):
    """
    Get the number of bytes currently allocated by the
    OperatorBufferPool of this groupby operator.
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
            builder.module, fnty, name="groupby_get_op_pool_bytes_allocated"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.uint64(groupby_state)
    return sig, codegen


@intrinsic
def get_num_partitions(
    typingctx,
    groupby_state,
):
    """
    Get the number of partitions of this groupby operator.
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
            builder.module, fnty, name="groupby_get_num_partitions"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.uint32(groupby_state)
    return sig, codegen


@intrinsic
def get_partition_num_top_bits_by_idx(typingctx, groupby_state, idx):
    """
    Get the number of bits in the 'top_bitmask' of a partition of this groupby
    operator by the partition index.
    This is only used for testing and debugging purposes.
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(32),
            [lir.IntType(8).as_pointer(), lir.IntType(64)],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="groupby_get_partition_num_top_bits_by_idx"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.uint32(groupby_state, idx)
    return sig, codegen


@intrinsic
def get_partition_top_bitmask_by_idx(typingctx, groupby_state, idx):
    """
    Get the 'top_bitmask' of a partition of this groupby operator by the partition index.
    This is only used for testing and debugging purposes.
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(32),
            [lir.IntType(8).as_pointer(), lir.IntType(64)],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="groupby_get_partition_top_bitmask_by_idx"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.uint32(groupby_state, idx)
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_partition_state(groupby_state):
    """
    Get the partition state (the number of bits in the 'top_bitmask' and 'top_bitmask')
    of all partitions of this groupby operator.
    This is only used for testing and debugging purposes.
    """

    def impl(groupby_state):  # pragma: no cover
        partition_state = []
        for idx in range(get_num_partitions(groupby_state)):
            partition_state.append(
                (
                    get_partition_num_top_bits_by_idx(groupby_state, idx),
                    get_partition_top_bitmask_by_idx(groupby_state, idx),
                )
            )
        return partition_state

    return impl
