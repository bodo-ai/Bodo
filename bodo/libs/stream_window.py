# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Support for streaming window functions.
"""
import typing as pt
from functools import cached_property

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
from numba.extending import intrinsic, lower_builtin, models, overload, register_model

import bodo
from bodo.ext import stream_window_cpp
from bodo.hiframes.pd_groupby_ext import get_window_func_types
from bodo.ir.aggregate import supported_agg_funcs
from bodo.libs.array import (
    cpp_table_to_py_table,
    delete_table,
    py_data_to_cpp_table,
)
from bodo.libs.array import (
    table_type as cpp_table_type,
)
from bodo.libs.decimal_arr_ext import decimal_division_output_precision_scale
from bodo.libs.stream_base import StreamingStateType
from bodo.utils.transform import get_call_expr_arg
from bodo.utils.typing import (
    BodoError,
    MetaType,
    dtype_to_array_type,
    error_on_unsupported_streaming_arrays,
    get_common_bodosql_integer_arr_type,
    get_overload_const_int,
    is_bodosql_integer_arr_type,
    is_nullable,
    is_overload_none,
    to_nullable_type,
    unwrap_typeref,
)

ll.add_symbol(
    "window_state_init_py_entry", stream_window_cpp.window_state_init_py_entry
)
ll.add_symbol(
    "window_build_consume_batch_py_entry",
    stream_window_cpp.window_build_consume_batch_py_entry,
)
ll.add_symbol(
    "window_produce_output_batch_py_entry",
    stream_window_cpp.window_produce_output_batch_py_entry,
)
ll.add_symbol("delete_window_state", stream_window_cpp.delete_window_state)

null_array_type = bodo.libs.null_arr_ext.NullArrayType()


def scalar_to_arrtype(scalar):
    return dtype_to_array_type(bodo.typeof(scalar))


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
    func_input_indices: pt.Tuple[pt.Tuple[int, ...], ...]
    window_args: pt.Tuple[pt.Tuple[int, ...], ...]
    build_table_type: pt.Union[bodo.hiframes.table.TableType, types.unknown]

    def __init__(
        self,
        partition_indices,
        order_by_indices,
        is_ascending,
        nulls_last,
        func_names,
        func_input_indices,
        kept_input_indices,
        n_inputs,
        window_args,
        build_table_type=types.unknown,
    ):
        error_on_unsupported_streaming_arrays(build_table_type)

        self.partition_indices = partition_indices
        self.order_by_indices = order_by_indices
        self.is_ascending = is_ascending
        self.nulls_last = nulls_last
        self.func_names = func_names
        self.kept_input_indices = kept_input_indices
        self.kept_input_indices_set = set(kept_input_indices)
        self.func_input_indices = func_input_indices
        self.n_inputs = n_inputs
        self.window_args = window_args
        self.build_table_type = build_table_type
        super().__init__(
            name=f"WindowStateType({partition_indices=}, {order_by_indices=}, {is_ascending=}, {nulls_last=}, {func_names=}, {func_input_indices=}, {kept_input_indices=}, {n_inputs=}, {window_args=}, {build_table_type=})"
        )

    @staticmethod
    def derive_common_arr_types(arr_type1, arr_type2):
        common_arr_type = arr_type1
        if common_arr_type == null_array_type:
            return to_nullable_type(arr_type2)

        if arr_type2 == null_array_type:
            return to_nullable_type(common_arr_type)

        if is_nullable(arr_type2):
            common_arr_type = to_nullable_type(common_arr_type)

        if isinstance(common_arr_type, bodo.MapArrayType):
            assert isinstance(arr_type2, bodo.MapArrayType)
            common_key_type = WindowStateType.derive_common_arr_types(
                common_arr_type.key_arr_type, arr_type2.key_arr_type
            )
            common_value_type = WindowStateType.derive_common_arr_types(
                common_arr_type.value_arr_type, arr_type2.value_arr_type
            )
            return bodo.MapArrayType(common_key_type, common_value_type)

        if isinstance(common_arr_type, bodo.ArrayItemArrayType):
            assert isinstance(arr_type2, bodo.ArrayItemArrayType)
            common_element_type = WindowStateType.derive_common_arr_types(
                common_arr_type.dtype, arr_type2.dtype
            )
            return bodo.ArrayItemArrayType(common_element_type)

        if isinstance(common_arr_type, bodo.StructArrayType):
            assert isinstance(arr_type2, bodo.StructArrayType)
            n_fields = len(common_arr_type.data)
            field_names = common_arr_type.names
            assert len(arr_type2.data) == n_fields and arr_type2.names == field_names
            common_field_types = []
            for arr1_field, arr2_field in zip(common_arr_type.data, arr_type2.data):
                common_field_types.append(
                    WindowStateType.derive_common_arr_types(arr1_field, arr2_field)
                )
            return bodo.StructArrayType(tuple(common_field_types))

        valid_str_types = (bodo.string_array_type, bodo.dict_str_arr_type)
        if common_arr_type in valid_str_types:
            # if the input column is a dictionary keep it as a dict, and if it is a string keep it as string
            assert arr_type2 in valid_str_types
            return common_arr_type

        # integer, float, decimal array case
        # NOTE: here we are assuming that the types are "castable" e.g. Int and Int, Float and Float,
        # Decimal and Deciaml (with the same scale/precision)
        if is_bodosql_integer_arr_type(common_arr_type) or isinstance(
            common_arr_type.dtype, types.Float
        ):
            common_arr_type = get_common_bodosql_integer_arr_type(
                [common_arr_type, arr_type2]
            )

        # other cases we case the the default to the same type as the input
        return common_arr_type

    def derive_common_table_types(self):
        if self.build_table_type == types.unknown:
            return types.unknown, types.unknown

        build_table_arr_types = list(self.build_table_type.arr_types)

        args_arr_types = []

        for func_name, indices, args in zip(
            self.func_names, self.func_input_indices, self.window_args
        ):
            if func_name not in supported_agg_funcs:
                raise BodoError(func_name + " is not a supported aggregate function.")

            # we potentially need to do casting...
            if func_name in ["lead", "lag"]:
                input_arr_type = build_table_arr_types[indices[0]]
                shift_amt, default_value = args
                common_arr_type = WindowStateType.derive_common_arr_types(
                    input_arr_type, scalar_to_arrtype(default_value)
                )
                # cast the shift amt to an int64 numpy for consistency
                args_arr_types.append(scalar_to_arrtype(np.int64(0)))
                args_arr_types.append(common_arr_type)
                build_table_arr_types[indices[0]] = common_arr_type
            else:
                # otherwise just add the types of the args
                args_arr_types.extend(map(scalar_to_arrtype, args))

        if len(args_arr_types) == 0:
            args_arr_types.append(scalar_to_arrtype(None))

        return bodo.TableType(tuple(build_table_arr_types)), bodo.TableType(
            tuple(args_arr_types)
        )

    def translate_indices(self, indices: pt.List[int]) -> pt.List[int]:
        """
        Maps the integers in indices to the new column that they
        correspond to when the table is passed in to C++.
        """
        out = []
        n_partition_cols = len(self.partition_indices)
        n_order_cols = len(self.order_by_indices)
        for col_idx in indices:
            if col_idx in self.partition_indices:
                new_idx = self.partition_indices.index(col_idx)
            elif col_idx in self.order_by_indices:
                new_idx = self.order_by_indices.index(col_idx) + n_partition_cols
            else:
                n_partition_before = len(
                    [i for i in self.partition_indices if i < col_idx]
                )
                n_order_before = len([i for i in self.order_by_indices if i < col_idx])
                new_idx = (
                    (n_partition_cols + n_order_cols)
                    + col_idx
                    - (n_partition_before + n_order_before)
                )
            out.append(new_idx)

        return out

    @property
    def key(self):
        return (
            self.partition_indices,
            self.order_by_indices,
            self.is_ascending,
            self.nulls_last,
            self.func_names,
            self.func_input_indices,
            self.kept_input_indices,
            self.build_table_type,
            self.n_inputs,
            self.window_args,
        )

    @cached_property
    def is_sort_impl(self) -> bool:
        if bodo.bodo_disable_streaming_window_sort:
            return False
        return all(
            [name in self.sort_supporting_func_names() for name in self.func_names]
        )

    @staticmethod
    def sort_supporting_func_names():
        """Determine which function names can take the sort based path.

        Returns:
            Set[str]: Set of function names that can support sort.
        """
        return {
            "first",
            "last",
            "min",
            "max",
            "count",
            "count_if",
            "boolor_agg",
            "booland_agg",
            "bitor_agg",
            "bitand_agg",
            "bitxor_agg",
            "sum",
            "mean",
            "dense_rank",
            "row_number",
            "rank",
            "percent_rank",
            "cume_dist",
            "size",
        }

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
    def cols_to_keep(self) -> pt.List[bool]:
        """Converts kept_input_indices to a bitmask of length num_input_cols

        Returns:
            pt.List[bool]: The bitmask over all input columns where True indicates that we should keep it in the output.
        """
        result = [i in self.kept_input_indices_set for i in self.partition_indices]
        result.extend([i in self.kept_input_indices_set for i in self.order_by_indices])
        for i in range(self.n_inputs):
            if i in self.partition_indices or i in self.order_by_indices:
                continue
            result.append(i in self.kept_input_indices_set)
        return result

    @cached_property
    def partition_by_types(self) -> pt.List[types.ArrayCompatible]:
        """Generate the list of array types that should be used for the
        partition by keys.

        Returns:
            List[types.ArrayCompatible]: The list of array types used
            by partition by.
        """
        casted_build_table_type, _ = self.derive_common_table_types()
        if casted_build_table_type == types.unknown:
            # Typing transformations haven't fully finished yet.
            return []

        partition_indices = self.partition_indices
        arr_types = []
        num_keys = len(partition_indices)
        arr_types = [
            casted_build_table_type.arr_types[partition_indices[i]]
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
        casted_build_table_type, _ = self.derive_common_table_types()
        if casted_build_table_type == types.unknown:
            # Typing transformations haven't fully finished yet.
            return []
        order_by_indices = self.order_by_indices

        num_sort_cols = len(order_by_indices)
        arr_types = [
            casted_build_table_type.arr_types[order_by_indices[i]]
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
        table, _ = self.derive_common_table_types()
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

    def inputs_to_function(self, func_idx) -> pt.List[int]:
        """
        Get the indices of the input columns to the func_idx-th function call.
        """
        return list(self.func_input_indices[func_idx])

    @cached_property
    def out_table_type(self):
        if self.build_table_type == types.unknown:
            return types.unknown

        casted_build_table_type, _ = self.derive_common_table_types()

        # The output table puts all the input columns first, in the original order, followed by the window function outputs
        input_arr_types = [
            casted_build_table_type.arr_types[i] for i in self.kept_input_indices
        ]
        # Now include the output types for any window functions
        window_func_types = get_window_func_types()
        for func_idx, func_name in enumerate(self.func_names):
            if func_name in window_func_types:
                output_type = window_func_types[func_name]

                # for mean we want the output type to be float64 unless we are  in the decimal case
                if func_name == "mean":
                    indices = self.inputs_to_function(func_idx)

                    input_index = indices[0]
                    input_type = casted_build_table_type.arr_types[input_index]
                    in_dtype = input_type.dtype

                    # Here we use the typing rules for sum + division to derive the type for mean. This differs from Snowflake behavior: Snowflake adds 3 to the scale by default. If the input scale is >34 it gives an error
                    if isinstance(in_dtype, bodo.Decimal128Type):
                        out_p = bodo.libs.decimal_arr_ext.DECIMAL128_MAX_PRECISION
                        _, out_s = decimal_division_output_precision_scale(
                            out_p, in_dtype.scale, out_p, 0
                        )
                        out_dtype = bodo.Decimal128Type(
                            out_p,
                            out_s,
                        )
                        output_type = dtype_to_array_type(out_dtype)

                if output_type is None:
                    # None = infer from input column
                    indices = self.inputs_to_function(func_idx)
                    assert (
                        len(indices) == 1
                    ), f"Expected 1 input column to function {func_name}, received {len(indices)}"
                    input_index = indices[0]
                    input_type = casted_build_table_type.arr_types[input_index]
                    if func_name in {
                        "min",
                        "max",
                        "lead",
                        "lag",
                        "bitand_agg",
                        "bitor_agg",
                        "bitxor_agg",
                        "first",
                        "last",
                    }:
                        output_type = input_type
                    elif func_name == "sum":
                        in_dtype = input_type.dtype
                        if isinstance(in_dtype, bodo.Decimal128Type):
                            out_dtype = bodo.Decimal128Type(
                                bodo.libs.decimal_arr_ext.DECIMAL128_MAX_PRECISION,
                                in_dtype.scale,
                            )
                            output_type = dtype_to_array_type(out_dtype)
                        elif isinstance(in_dtype, types.Integer) and (
                            in_dtype.bitwidth <= 64
                        ):
                            # Upcast output integer to the 64-bit variant to prevent overflow.
                            out_dtype = types.int64 if in_dtype.signed else types.uint64
                            if isinstance(input_type, types.Array):
                                # If regular numpy (i.e. non-nullable)
                                output_type = dtype_to_array_type(out_dtype)
                            else:
                                # Nullable:
                                output_type = dtype_to_array_type(
                                    out_dtype, convert_nullable=True
                                )
                        else:
                            output_type = input_type
                    else:
                        raise BodoError(
                            func_name + " is not a supported window function."
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


@intrinsic
def _init_window_state(
    typingctx,
    operator_id,
    build_arr_dtypes,
    build_arr_array_types,
    n_build_arrs,
    window_ftypes_t,
    n_funcs_t,
    order_by_asc_t,
    order_by_na_t,
    n_order_by_keys_t,
    cols_to_keep_t,
    n_input_cols_t,
    func_input_indices_t,
    func_input_offsets_t,
    output_state_type,
    parallel_t,
    allow_work_stealing_t,
):
    """Initialize C++ WindowState pointer

    Args:
        operator_id (int64): ID of this operator (used for looking up budget),
        build_arr_dtypes (int8*): pointer to array of ints representing array dtypes
                                   (as provided by numba_to_c_type)
        build_arr_array_types (int8*): pointer to array of ints representing array types
                                    (as provided by numba_to_c_array_type)
        n_build_arrs (int32): number of build columns
        ftypes (int32*): List of window functions to use
        n_funcs (int): Number of window functions.
        order_by_asc_t (bool*): Bitmask for sort direction of order-by columns.
        order_by_na_t (bool*): Bitmask for null sort direction of order-by columns.
        func_input_indices_t (int*): List of indices for window function input columns.
        func_input_offsets_t (int*): List of offsets mapping each window function to the
        subset of func_input_indices_t that it corresponds to.
        n_order_by_keys_t (int): Number of order-by columns.
        partition_by_cols_to_keep_t (bool*): Bitmask of partition/key columns to retain in output.
        order_by_cols_to_keep_t (bool*): Bitmask of order-by columns to retain in output.
        input_cols_to_keep_t (bool*): Bitmask of input columns to retain in output.
        n_input_cols_to_keep_t (bool*): Number if input columns.
        output_state_type (TypeRef[WindowStateType]): The output type for the state
                                                    that should be generated.
        parallel_t (bool): Is this executed in parallel.
        allow_work_stealing_t (bool): Is work stealing allowed?
    """
    output_type = unwrap_typeref(output_state_type)

    def codegen(context, builder, sig, args):
        (
            operator_id,
            build_arr_dtypes,
            build_arr_array_types,
            n_build_arrs,
            window_ftypes,
            n_funcs,
            order_by_asc,
            order_by_na,
            n_order_by_keys,
            cols_to_keep,
            n_input_cols,
            func_input_indices,
            func_input_offsets,
            _,  # output_state_type
            parallel,
            allow_work_stealing,
        ) = args
        n_keys = context.get_constant(types.uint64, output_type.n_keys)
        output_batch_size = context.get_constant(
            types.int64, bodo.bodosql_streaming_batch_size
        )
        # We don't support adaptive shuffling yet.
        stream_loop_sync_iters = (
            bodo.default_stream_loop_sync_iters
            if bodo.stream_loop_sync_iters == -1
            else bodo.stream_loop_sync_iters
        )
        sync_iter = context.get_constant(types.int64, stream_loop_sync_iters)
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(32).as_pointer(),
                lir.IntType(32),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(32).as_pointer(),
                lir.IntType(32).as_pointer(),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(64),
                lir.IntType(1),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="window_state_init_py_entry"
        )
        input_args = (
            operator_id,
            build_arr_dtypes,
            build_arr_array_types,
            n_build_arrs,
            window_ftypes,
            n_funcs,
            n_keys,
            order_by_asc,
            order_by_na,
            n_order_by_keys,
            cols_to_keep,
            n_input_cols,
            func_input_indices,
            func_input_offsets,
            output_batch_size,
            parallel,
            sync_iter,
            allow_work_stealing,
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
        types.int32,
        types.CPointer(types.bool_),
        types.CPointer(types.bool_),
        types.int64,
        types.CPointer(types.bool_),
        types.int32,
        types.CPointer(types.int32),
        types.CPointer(types.int32),
        output_state_type,
        parallel_t,
        allow_work_stealing_t,
    )
    return sig, codegen


def init_window_state(
    operator_id,
    partition_indices,
    order_by_indices,
    is_ascending,
    nulls_last,
    func_names,
    func_input_indices,
    kept_input_indices,
    allow_work_stealing,
    n_inputs,
    window_args,
    op_pool_size_bytes=-1,
    expected_state_type=None,
    parallel=False,
):
    pass


@infer_global(init_window_state)
class InitWindowStateInfer(AbstractTemplate):
    """Typer for init_window_state that returns state type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(init_window_state)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        expected_state_type = unwrap_typeref(folded_args[12])
        (
            partition_indices,
            order_by_indices,
            is_ascending,
            nulls_last,
            func_names,
            func_input_indices,
            kept_input_indices,
            _,
            n_inputs,
            window_args,
        ) = folded_args[1:11]
        if is_overload_none(expected_state_type):
            partition_indices_tuple = unwrap_typeref(partition_indices).meta
            order_by_indices_tuple = unwrap_typeref(order_by_indices).meta
            is_ascending_tuple = unwrap_typeref(is_ascending).meta
            nulls_last_tuple = unwrap_typeref(nulls_last).meta
            func_names_tuple = unwrap_typeref(func_names).meta
            func_input_tuple = unwrap_typeref(func_input_indices).meta
            kept_input_indices_tuple = unwrap_typeref(kept_input_indices).meta
            window_args = unwrap_typeref(window_args).meta
            output_type = WindowStateType(
                partition_indices_tuple,
                order_by_indices_tuple,
                is_ascending_tuple,
                nulls_last_tuple,
                func_names_tuple,
                func_input_tuple,
                kept_input_indices_tuple,
                get_overload_const_int(n_inputs),
                window_args,
            )
        else:
            output_type = expected_state_type
        return signature(output_type, *folded_args).replace(pysig=pysig)


InitWindowStateInfer._no_unliteral = True


@lower_builtin(init_window_state, types.VarArg(types.Any))
def lower_init_window_state(context, builder, sig, args):
    """lower init_window_state() using overload_init_window_state"""
    impl = overload_init_window_state(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def overload_init_window_state(
    operator_id,
    partition_indices,
    order_by_indices,
    is_ascending,
    nulls_last,
    func_names,
    func_input_indices,
    kept_input_indices,
    allow_work_stealing,
    n_inputs,
    window_args,
    op_pool_size_bytes=-1,
    expected_state_type=None,
    parallel=False,
):
    output_type: WindowStateType = unwrap_typeref(expected_state_type)  # type: ignore

    build_arr_dtypes = output_type.build_arr_ctypes
    build_arr_array_types = output_type.build_arr_array_types
    n_build_arrs = len(build_arr_dtypes)
    ftypes = [supported_agg_funcs.index("window")]
    window_ftypes = []
    for fname in output_type.func_names:
        if fname not in supported_agg_funcs:
            raise BodoError(fname + " is not a supported aggregate function.")
        window_ftypes.append(supported_agg_funcs.index(fname))
    ftypes_arr = np.array(ftypes, np.int32)
    window_ftypes_arr = np.array(window_ftypes, np.int32)
    n_funcs = len(output_type.func_names)
    sort_ascending_arr = np.array(output_type.is_ascending, np.bool_)
    sort_nulls_last_arr = np.array(output_type.nulls_last, np.bool_)

    kept_cols_arr = np.array(output_type.cols_to_keep, np.bool_)

    n_input_cols = np.int32(
        len(output_type.cols_to_keep)
        - len(output_type.partition_indices)
        - len(output_type.order_by_indices)
    )
    n_orderby_cols = len(output_type.order_by_indices)
    func_input_indices_list = []
    func_input_offsets_list = [0]
    for indices in output_type.func_input_indices:
        func_input_indices_list += output_type.translate_indices(indices)
        func_input_offsets_list.append(len(func_input_indices_list))
    func_input_indices_arr = np.array(func_input_indices_list, np.int32)
    func_input_offsets_arr = np.array(func_input_offsets_list, np.int32)
    window_args_list = []
    for window_args in output_type.window_args:
        # TODO: verify number of scalar arguments for each window function
        window_args_list.extend(window_args)
    # NOTE: Adds None for funcs with no args so that it compiles
    if len(window_args) == 0:
        window_args_list.append(None)

    window_args_tuple = tuple(window_args_list)
    _, scalar_args_table = output_type.derive_common_table_types()
    scalar_args_arr_types = scalar_args_table.arr_types
    in_col_inds = MetaType(tuple(range(len(window_args_tuple))))

    if output_type.is_sort_impl:
        # The internal C++ window state object is just for sort implementations
        # right now.
        def impl(
            operator_id,
            partition_indices,
            order_by_indices,
            is_ascending,
            nulls_last,
            func_names,
            func_input_indices,
            kept_input_indices,
            allow_work_stealing,
            n_inputs,
            window_args,
            op_pool_size_bytes=-1,
            expected_state_type=None,
            parallel=False,
        ):  # pragma: no cover
            output_val = bodo.libs.stream_window._init_window_state(
                operator_id,
                build_arr_dtypes.ctypes,
                build_arr_array_types.ctypes,
                n_build_arrs,
                window_ftypes_arr.ctypes,
                n_funcs,
                sort_ascending_arr.ctypes,
                sort_nulls_last_arr.ctypes,
                n_orderby_cols,
                kept_cols_arr.ctypes,
                n_input_cols,
                func_input_indices_arr.ctypes,
                func_input_offsets_arr.ctypes,
                output_type,
                parallel,
                allow_work_stealing,
            )
            return output_val

        return impl
    else:
        if len(output_type.partition_indices) == 0:  # pragma: no cover
            raise BodoError(
                "Invalid window state: cannot use hash-based implementation of window functions without partition columns"
            )
        # The hash partition window state C++ object is just a group by state.
        func_text = """
def impl(
        operator_id,
        partition_indices,
        order_by_indices,
        is_ascending,
        nulls_last,
        func_names,
        func_input_indices,
        kept_input_indices,
        allow_work_stealing,
        n_inputs,
        window_args,  # list of tuples containing scalar window args
        op_pool_size_bytes=-1,
        expected_state_type=None,
        parallel=False,
):\n
"""
        func_text += "    window_args_arrs = ({},)\n".format(
            ",".join(
                [
                    "bodo.utils.conversion.coerce_scalar_to_array(window_args_tuple[{}], 1, types.unknown)\n".format(
                        i
                    )
                    for i in range(len(window_args_tuple))
                ]
            )
        )
        func_text += "    window_args_table = bodo.hiframes.table.logical_table_to_table((), tuple(window_args_arrs), in_col_inds, 0)\n"
        func_text += "    casted_args_table = bodo.utils.table_utils.table_astype(window_args_table, scalar_args_table, False, False)\n"
        func_text += "    window_args_cpp_table = bodo.libs.array.py_data_to_cpp_table(casted_args_table, (), in_col_inds, {})\n".format(
            len(window_args_tuple)
        )
        func_text += """
    output_val = bodo.libs.stream_groupby._init_groupby_state(
        operator_id,
        build_arr_dtypes.ctypes,
        build_arr_array_types.ctypes,
        n_build_arrs,
        ftypes_arr.ctypes,
        window_ftypes_arr.ctypes,
        func_input_offsets_arr.ctypes,
        func_input_indices_arr.ctypes,
        n_funcs,
        sort_ascending_arr.ctypes,
        sort_nulls_last_arr.ctypes,
        n_orderby_cols,
        kept_cols_arr.ctypes,
        window_args_cpp_table,
        op_pool_size_bytes,
        output_type,
        parallel,
    )
    return output_val
"""
        local_vars = {}
        global_vars = {
            "bodo": bodo,
            "types": types,
            "build_arr_dtypes": build_arr_dtypes,
            "build_arr_array_types": build_arr_array_types,
            "n_build_arrs": n_build_arrs,
            "ftypes_arr": ftypes_arr,
            "window_ftypes_arr": window_ftypes_arr,
            "func_input_offsets_arr": func_input_offsets_arr,
            "func_input_indices_arr": func_input_indices_arr,
            "n_funcs": n_funcs,
            "sort_ascending_arr": sort_ascending_arr,
            "sort_nulls_last_arr": sort_nulls_last_arr,
            "n_orderby_cols": n_orderby_cols,
            "window_args_tuple": window_args_tuple,
            "scalar_args_arr_types": scalar_args_arr_types,
            "output_type": output_type,
            "scalar_args_table": scalar_args_table,
            "in_col_inds": in_col_inds,
            "kept_cols_arr": kept_cols_arr,
        }
        exec(func_text, global_vars, local_vars)
        impl = local_vars["impl"]
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
        tuple(bool, bool): is last batch globally with possibility of false negatives
        due to iterations between syncs, request new input flag (always true)
    """
    in_col_inds = MetaType(window_state.build_indices)
    n_table_cols = len(in_col_inds)
    casted_table_type, _ = window_state.derive_common_table_types()

    if window_state.is_sort_impl:

        def impl_window_build_consume_batch(
            window_state, table, is_last
        ):  # pragma: no cover
            cpp_table = py_data_to_cpp_table(table, (), in_col_inds, n_table_cols)
            # Currently the window state C++ object is just a group by state.
            is_last = bodo.libs.stream_window._window_build_consume_batch(
                window_state, cpp_table, is_last
            )
            # Returning True for input request flag since there are no shuffle buffers
            # and we just accumulate input in this case.
            return is_last, True

        return impl_window_build_consume_batch

    else:

        def impl_window_build_consume_batch(
            window_state, table, is_last
        ):  # pragma: no cover
            casted_table = bodo.utils.table_utils.table_astype(
                table, casted_table_type, False, False
            )

            cpp_table = py_data_to_cpp_table(
                casted_table, (), in_col_inds, n_table_cols
            )
            # Currently the window state C++ object is just a group by state.
            return bodo.libs.stream_groupby._groupby_build_consume_batch(
                window_state, cpp_table, is_last, True
            )

        return impl_window_build_consume_batch


@intrinsic
def _window_build_consume_batch(
    typingctx,
    groupby_state,
    cpp_table,
    is_last,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(1),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="window_build_consume_batch_py_entry"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.bool_(groupby_state, cpp_table, is_last)
    return sig, codegen


@infer_global(window_build_consume_batch)
class WindowBuildConsumeBatchInfer(AbstractTemplate):
    """Typer for groupby_build_consume_batch that returns bool as output type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(window_build_consume_batch)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        output_type = types.BaseTuple.from_types((types.bool_, types.bool_))
        return signature(output_type, *folded_args).replace(pysig=pysig)


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

    out_cols = window_state.cpp_output_table_to_py_table_indices
    out_cols_arr = np.array(out_cols, dtype=np.int64)

    if window_state.is_sort_impl:

        def impl_window_produce_output_batch(
            window_state,
            produce_output,
        ):  # pragma: no cover
            (
                out_cpp_table,
                out_is_last,
            ) = bodo.libs.stream_window._window_produce_output_batch(
                window_state, produce_output
            )
            out_table = cpp_table_to_py_table(
                out_cpp_table, out_cols_arr, out_table_type, 0
            )
            delete_table(out_cpp_table)
            return out_table, out_is_last

        return impl_window_produce_output_batch

    else:

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


@intrinsic
def _window_produce_output_batch(
    typingctx,
    window_state,
    produce_output,
):
    def codegen(context, builder, sig, args):
        out_is_last = cgutils.alloca_once(builder, lir.IntType(1))
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [lir.IntType(8).as_pointer(), lir.IntType(1).as_pointer(), lir.IntType(1)],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="window_produce_output_batch_py_entry"
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
        window_state,
        produce_output,
    )
    return sig, codegen


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
        StreamingStateType.ensure_known_inputs(
            "window_produce_output_batch",
            (window_state.build_table_type,),
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

    if window_state.is_sort_impl:
        return lambda window_state: bodo.libs.stream_window._delete_window_state(
            window_state
        )
    else:
        # Currently the window state C++ object is just a group by state.
        return lambda window_state: bodo.libs.stream_groupby.delete_groupby_state(
            window_state
        )  # pragma: no cover


@intrinsic
def _delete_window_state(
    typingctx,
    window_state,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="delete_window_state"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    sig = types.void(window_state)
    return sig, codegen
