# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Support for streaming window functions.
"""
import typing as pt
from functools import cached_property

import numba
import numpy as np
from numba.core import types
from numba.core.typing.templates import (
    AbstractTemplate,
    infer_global,
    signature,
)
from numba.extending import lower_builtin, models, overload, register_model

import bodo
from bodo.hiframes.pd_groupby_ext import get_window_func_types
from bodo.ir.aggregate import supported_agg_funcs
from bodo.libs.array import (
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
    is_overload_none,
    unwrap_typeref,
)


class WindowStateType(StreamingStateType):
    """Type for a C++ window state pointer. Currently
    this is a wrapper around the aggregate state with
    some additional configuration."""

    partition_indices: pt.Tuple[int, ...]
    order_by_indices: pt.Tuple[int, ...]
    is_ascending: pt.Tuple[bool, ...]
    nulls_last: pt.Tuple[bool, ...]
    func_names: pt.Tuple[str, ...]
    kept_input_indices: pt.Tuple[int, ...]
    kept_input_indices_set: pt.Set[int]
    build_table_type: pt.Union[bodo.hiframes.table.TableType, types.unknown]

    def __init__(
        self,
        partition_indices,
        order_by_indices,
        is_ascending,
        nulls_last,
        func_names,
        kept_input_indices,
        build_table_type=types.unknown,
    ):
        error_on_unsupported_streaming_arrays(build_table_type)

        self.partition_indices = partition_indices
        self.order_by_indices = order_by_indices
        self.is_ascending = is_ascending
        self.nulls_last = nulls_last
        self.func_names = func_names
        if len(func_names) != 1:
            raise BodoError("Streaming Window only supports a single function.")
        self.kept_input_indices = kept_input_indices
        self.kept_input_indices_set = set(kept_input_indices)
        self.build_table_type = build_table_type
        super().__init__(
            name=f"WindowStateType({partition_indices=}, {order_by_indices=}, {is_ascending=}, {nulls_last=}, {func_names=}, {kept_input_indices=}, {build_table_type=})"
        )

    @property
    def key(self):
        return (
            self.partition_indices,
            self.order_by_indices,
            self.is_ascending,
            self.nulls_last,
            self.func_names,
            self.build_table_type,
        )

    @staticmethod
    def _derive_input_type(
        partition_by_types: list[types.ArrayCompatible],
        partition_by_indices: tuple[int],
        order_by_types: list[types.ArrayCompatible],
        order_by_indices: tuple[int],
        table_type: bodo.hiframes.table.TableType,
    ) -> pt.List[types.ArrayCompatible]:
        """Generate the input table type based on the type and indices information.

        Args:
            partition_by_types (List[types.ArrayCompatible]): The list of partition by types in order.
            partition_by_indices (N Tuple(int)): The indices of the partition by columns.
            order_by_types (List[types.ArrayCompatible]): The list of order by column types in order.
            order_by_indices (N Tuple(int)): The indices of the order by columns.

        Returns:
            List[types.ArrayCompatible]: The list of array types for the input C++ table (in order).
        """

        # The columns are: [<partition by columns>, <order by columns>, <rest of the columns>]
        types = partition_by_types + order_by_types
        idx_set = set(list(partition_by_indices) + list(order_by_indices))

        # Append the data columns
        for i in range(len(table_type.arr_types)):
            if i not in idx_set:
                types.append(table_type.arr_types[i])
        return types

    @cached_property
    def kept_partition_by_cols(self) -> pt.List[bool]:
        """
        Get the indices of the partition by columns that are kept as input.
        """
        return [i in self.kept_input_indices_set for i in self.partition_indices]

    @cached_property
    def kept_order_by_cols(self) -> pt.List[bool]:
        """
        Get the indices of the partition by columns that are kept as input.
        """
        return [i in self.kept_input_indices_set for i in self.order_by_indices]

    @cached_property
    def partition_by_types(self) -> pt.List[types.ArrayCompatible]:
        """Generate the list of array types that should be used for the
        partition by keys.

        Returns:
            List[types.ArrayCompatible]: The list of array types used
            by partition by.
        """
        build_table_type = self.build_table_type
        if build_table_type == types.unknown:
            # Typing transformations haven't fully finished yet.
            return []

        partition_indices = self.partition_indices
        arr_types = []
        num_keys = len(partition_indices)
        arr_types = [
            self.build_table_type.arr_types[partition_indices[i]]
            for i in range(num_keys)
        ]

        return arr_types

    @cached_property
    def order_by_types(self) -> list[types.ArrayCompatible]:
        """Generate the list of array types that should be used for the
        order by keys.

        Returns:
            List[types.ArrayCompatible]: The list of array types used
            by order by.
        """
        build_table_type = self.build_table_type
        if build_table_type == types.unknown:
            # Typing transformations haven't fully finished yet.
            return []
        order_by_indices = self.order_by_indices

        num_sort_cols = len(order_by_indices)
        arr_types = [
            self.build_table_type.arr_types[order_by_indices[i]]
            for i in range(num_sort_cols)
        ]

        return arr_types

    @cached_property
    def build_reordered_arr_types(self) -> pt.List[types.ArrayCompatible]:
        """
        Get the list of array types for the actual input to the C++ build table.
        This is different from the build_table_type because the input to the C++
        will reorder partition by columns in the front, followed by any order by
        columns. The order by columns will maintain the required sort order.

        Returns:
            List[types.ArrayCompatible]: The list of array types for the build table.
        """
        if self.build_table_type == types.unknown:
            return []

        partition_by_types = self.partition_by_types
        partition_indices = self.partition_indices
        order_by_types = self.order_by_types
        order_by_indices = self.order_by_indices
        table = self.build_table_type
        return self._derive_input_type(
            partition_by_types,
            order_by_indices,
            order_by_types,
            partition_indices,
            table,
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

    @property
    def f_in_cols(self) -> pt.List[int]:
        """
        Get the indices that are treated as function inputs. Since we don't support
        arguments yet and achieve shuffle by treating all columns as function inputs,
        this is just the range of non-partition by columns
        """
        return list(
            range(len(self.partition_indices), len(self.build_reordered_arr_types))
        )

    @cached_property
    def out_table_type(self):
        if self.build_table_type == types.unknown:
            return types.unknown

        # The output table puts all the input columns first, in the original order, followed by the window function outputs
        input_arr_types = [
            self.build_table_type.arr_types[i] for i in self.kept_input_indices
        ]
        # Now include the output types for any window functions
        window_func_types = get_window_func_types()
        for func_name in self.func_names:
            if func_name in window_func_types:
                output_type = window_func_types[func_name]
                if output_type is None:
                    raise BodoError(
                        func_name
                        + " does not have an output type. This function is not supported with streaming."
                    )
                input_arr_types.append(output_type)
            else:
                raise BodoError(func_name + " is not a supported window function.")

        return bodo.TableType(tuple(input_arr_types))

    @staticmethod
    def _derive_cpp_indices(partition_indices, order_by_indices, num_cols):
        """Generate the indices used for the C++ table from the
        given Python table.

        Args:
            partition_indices (N Tuple(int)): The indices of the partition by columns.
            order_by_indices (tuple[int]): The indices of the order by columns.
            num_cols (int): The number of total columns in the table.

        Returns:
            N Tuple(int): Tuple giving the order of the output indices
        """
        total_idxs = list(partition_indices + order_by_indices)
        idx_set = set(list(partition_indices) + list(order_by_indices))
        for i in range(num_cols):
            if i not in idx_set:
                total_idxs.append(i)
        return tuple(total_idxs)

    @cached_property
    def build_indices(self):
        if self.build_table_type == types.unknown:
            return ()

        return self._derive_cpp_indices(
            self.partition_indices,
            self.order_by_indices,
            len(self.build_table_type.arr_types),
        )

    @cached_property
    def cpp_output_table_to_py_table_indices(self) -> pt.List[int]:
        """
        Generate the remapping to convert the C++ output table to its corresponding Python table.
        The C++ input is of the form (partition by, order by, rest of the columns, window columns).
        The Python table needs to remap this to be (original input order, window columns).

        What makes this slightly more complicated is that members of the partition by and order by columns
        may have been dropped, so we need to account for that.

        Returns:
            pt.List[int]: A list of py_output index for each column in the corresponding C++ location.
        """
        # Use kept_input_indices to generate a mapping from original index to its output index
        input_map = {}
        num_kept_columns = 0
        for idx in self.build_indices:
            # If an input is not found in input_map it must be dropped. Otherwise,
            # the build_indices order matches the C++ output.
            if idx in self.kept_input_indices_set:
                input_map[idx] = num_kept_columns
                num_kept_columns += 1
        output_indices = [input_map[idx] for idx in self.kept_input_indices]
        for _ in self.func_names:
            output_indices.append(len(output_indices))
        return output_indices

    @property
    def n_keys(self) -> int:
        """
        Number of keys in UNION DISTINCT case
        Intended for GroupBy Compatibility, otherwise use n_cols
        """
        return len(self.partition_indices)


register_model(WindowStateType)(models.OpaqueModel)


def init_window_state(
    operator_id,
    partition_indices,
    order_by_indices,
    is_ascending,
    nulls_last,
    func_names,
    kept_input_indices,
    expected_state_type=None,
    parallel=False,
):
    pass


@overload(init_window_state)
def overload_init_window_state(
    operator_id,
    partition_indices,
    order_by_indices,
    is_ascending,
    nulls_last,
    func_names,
    kept_input_indices,
    expected_state_type=None,
    parallel=False,
):
    expected_state_type: pt.Optional[WindowStateType] = unwrap_typeref(expected_state_type)  # type: ignore
    if is_overload_none(expected_state_type):
        partition_indices_tuple = unwrap_typeref(partition_indices).meta
        order_by_indices_tuple = unwrap_typeref(order_by_indices).meta
        is_ascending_tuple = unwrap_typeref(is_ascending).meta
        nulls_last_tuple = unwrap_typeref(nulls_last).meta
        func_names_tuple = unwrap_typeref(func_names).meta
        kept_input_indices_tuple = unwrap_typeref(kept_input_indices).meta
        output_type = WindowStateType(
            partition_indices_tuple,
            order_by_indices_tuple,
            is_ascending_tuple,
            nulls_last_tuple,
            func_names_tuple,
            kept_input_indices_tuple,
        )
    else:
        output_type = expected_state_type

    build_arr_dtypes = output_type.build_arr_ctypes
    build_arr_array_types = output_type.build_arr_array_types
    n_build_arrs = len(build_arr_dtypes)
    ftypes = [supported_agg_funcs.index("window")] * len(output_type.func_names)
    window_ftypes = []
    for fname in output_type.func_names:
        if fname not in supported_agg_funcs:
            raise BodoError(fname + " is not a supported aggregate function.")
        window_ftypes.append(supported_agg_funcs.index(fname))
    ftypes_arr = np.array(ftypes, np.int32)
    window_ftypes_arr = np.array(window_ftypes, np.int32)
    f_in_cols = output_type.f_in_cols
    f_in_offsets_arr = np.array([0, len(f_in_cols)], np.int32)
    f_in_cols_arr = np.array(f_in_cols, np.int32)
    n_funcs = len(output_type.func_names)
    sort_ascending_arr = np.array(output_type.is_ascending, np.bool_)
    sort_nulls_last_arr = np.array(output_type.nulls_last, np.bool_)
    kept_partition_cols_arr = np.array(output_type.kept_partition_by_cols, np.bool_)
    kept_order_by_cols_arr = np.array(output_type.kept_order_by_cols, np.bool_)

    n_orderby_cols = len(output_type.order_by_indices)

    def impl(
        operator_id,
        partition_indices,
        order_by_indices,
        is_ascending,
        nulls_last,
        func_names,
        kept_input_indices,
        expected_state_type=None,
        parallel=False,
    ):  # pragma: no cover
        # Currently the window state C++ object is just a group by state.
        output_val = bodo.libs.stream_groupby._init_groupby_state(
            operator_id,
            build_arr_dtypes.ctypes,
            build_arr_array_types.ctypes,
            n_build_arrs,
            ftypes_arr.ctypes,
            window_ftypes_arr.ctypes,
            f_in_offsets_arr.ctypes,
            f_in_cols_arr.ctypes,
            n_funcs,
            sort_ascending_arr.ctypes,
            sort_nulls_last_arr.ctypes,
            n_orderby_cols,
            kept_partition_cols_arr.ctypes,
            kept_order_by_cols_arr.ctypes,
            -1,  # op_pool_size_bytes
            output_type,
            parallel,
        )
        return output_val

    return impl


def window_build_consume_batch(window_state, table, is_last):
    pass


def gen_window_build_consume_batch_impl(window_state: WindowStateType, table, is_last):
    """Consume a build table batch in streaming window insert into the accumulate step
    based on the partitions.

    Args:
        window_state (WindowState): C++ WindowState pointer
        table (table_type): build table batch
        is_last (bool): is last batch (in this pipeline) locally
    Returns:
        bool: is last batch globally with possibility of false negatives due to iterations between syncs
    """
    in_col_inds = MetaType(window_state.build_indices)
    n_table_cols = len(in_col_inds)

    def impl_window_build_consume_batch(
        window_state, table, is_last
    ):  # pragma: no cover
        cpp_table = py_data_to_cpp_table(table, (), in_col_inds, n_table_cols)
        # Currently the window state C++ object is just a group by state.
        return bodo.libs.stream_groupby._groupby_build_consume_batch(
            window_state, cpp_table, is_last, True
        )

    return impl_window_build_consume_batch


@infer_global(window_build_consume_batch)
class WindowBuildConsumeBatchInfer(AbstractTemplate):
    """Typer for groupby_build_consume_batch that returns bool as output type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(window_build_consume_batch)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(types.bool_, *folded_args).replace(pysig=pysig)


@lower_builtin(window_build_consume_batch, types.VarArg(types.Any))
def lower_window_build_consume_batch(context, builder, sig, args):
    """lower window_build_consume_batch() using gen_window_build_consume_batch_impl above"""
    impl = gen_window_build_consume_batch_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def window_produce_output_batch(window_state, produce_output):
    pass


def gen_window_produce_output_batch_impl(window_state: WindowStateType, produce_output):
    """Produce output batches of the window operation

    Args:
        window_state (WindowStateType): C++ WindowState pointer
        produce_output (bool): whether to produce output

    Returns:
        table_type: output table batch
        bool: global is last batch with possibility of false negatives due to iterations between syncs
    """
    out_table_type = window_state.out_table_type

    if out_table_type == types.unknown:
        out_cols_arr = np.array([], dtype=np.int64)
    else:
        out_cols = window_state.cpp_output_table_to_py_table_indices
        out_cols_arr = np.array(out_cols, dtype=np.int64)

    def impl_window_produce_output_batch(
        window_state,
        produce_output,
    ):  # pragma: no cover
        # Currently the window state C++ object is just a group by state.
        (
            out_cpp_table,
            out_is_last,
        ) = bodo.libs.stream_groupby._groupby_produce_output_batch(
            window_state, produce_output
        )
        out_table = cpp_table_to_py_table(
            out_cpp_table, out_cols_arr, out_table_type, 0
        )
        delete_table(out_cpp_table)
        return out_table, out_is_last

    return impl_window_produce_output_batch


@infer_global(window_produce_output_batch)
class GroupbyProduceOutputInfer(AbstractTemplate):
    """Typer for window_produce_output_batch that returns (output_table_type, bool)
    as output type.
    """

    def generic(self, args, kws):
        kws = dict(kws)
        window_state = get_call_expr_arg(
            "window_produce_output_batch", args, kws, 0, "window_state"
        )
        out_table_type = window_state.out_table_type
        # Output is (out_table, out_is_last)
        output_type = types.BaseTuple.from_types((out_table_type, types.bool_))

        pysig = numba.core.utils.pysignature(window_produce_output_batch)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(output_type, *folded_args).replace(pysig=pysig)


@lower_builtin(window_produce_output_batch, types.VarArg(types.Any))
def lower_window_produce_output_batch(context, builder, sig, args):
    """lower window_produce_output_batch() using gen_window_produce_output_batch_impl above"""
    impl = gen_window_produce_output_batch_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def delete_window_state(window_state):
    pass


@overload(delete_window_state)
def overload_delete_window_state(window_state):
    if not isinstance(window_state, WindowStateType):  # pragma: no cover
        raise BodoError(
            f"delete_window_state: Expected type WindowStateType "
            f"for first arg `window_state`, found {window_state}"
        )

    # Currently the window state C++ object is just a group by state.
    return lambda window_state: bodo.libs.stream_groupby.delete_groupby_state(
        window_state
    )  # pragma: no cover