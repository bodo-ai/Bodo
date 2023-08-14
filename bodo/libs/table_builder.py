# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Interface to C++ TableBuilderState"""
from functools import cached_property
from typing import List

import llvmlite.binding as ll
import numba
import numpy as np
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic, models, register_model

import bodo
from bodo.ext import table_builder_cpp
from bodo.libs.array import (
    cpp_table_to_py_table,
    delete_table,
    py_data_to_cpp_table,
)
from bodo.libs.array import table_type as cpp_table_type
from bodo.utils.typing import (
    MetaType,
    error_on_nested_arrays,
    is_overload_none,
    unwrap_typeref,
)
from bodo.utils.utils import numba_to_c_array_type, numba_to_c_type

ll.add_symbol(
    "table_builder_state_init_py_entry",
    table_builder_cpp.table_builder_state_init_py_entry,
)
ll.add_symbol(
    "table_builder_append_py_entry",
    table_builder_cpp.table_builder_append_py_entry,
)
ll.add_symbol(
    "table_builder_finalize",
    table_builder_cpp.table_builder_finalize,
)


class TableBuilderStateType(types.Type):
    """Type for C++ TableBuilderState pointer"""

    def __init__(
        self,
        build_table_type=types.unknown,
    ):
        # TODO[BSE-937]: support nested arrays in streaming
        error_on_nested_arrays(build_table_type)
        self.build_table_type = build_table_type
        super().__init__(f"TableBuilderStateType(build_table={build_table_type})")

    @staticmethod
    def _derive_c_types(arr_types: List[types.ArrayCompatible]) -> np.ndarray:
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

    @cached_property
    def arr_dtypes(self) -> List[types.ArrayCompatible]:
        """Returns the list of types for each array in the build table."""
        ctypes = []
        if self.build_table_type != types.unknown:
            ctypes = self.build_table_type.arr_types
        return ctypes

    @cached_property
    def arr_ctypes(self) -> np.ndarray:
        return self._derive_c_types(self.arr_dtypes)

    @staticmethod
    def _derive_c_array_types(arr_types: List[types.ArrayCompatible]) -> np.ndarray:
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
    def arr_array_types(self) -> np.ndarray:
        """
        Fetch the CArrayTypeEnum used for each array in the build table.

        Returns:
            List(int): The CArrayTypeEnum for each array in the build table. Note
                that C++ wants the actual integer but these are the values derived from
                CArrayTypeEnum.
        """
        return self._derive_c_array_types(self.arr_dtypes)

    @property
    def num_input_arrs(self) -> int:
        """
        Determine the actual number of build arrays in the input.

        Return (int): The number of build arrays
        """
        return len(self.arr_ctypes)


register_model(TableBuilderStateType)(models.OpaqueModel)


@intrinsic
def _init_table_builder_state(
    typingctx, arr_ctypes, arr_array_ctypes, n_arrs, output_state_type
):
    output_type = unwrap_typeref(output_state_type)

    def codegen(context, builder, sig, args):
        (
            arr_ctypes,
            arr_array_ctypes,
            n_arrs,
            _,
        ) = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="table_builder_state_init_py_entry"
        )
        ret = builder.call(fn_tp, (arr_ctypes, arr_array_ctypes, n_arrs))
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = output_type(
        types.voidptr,
        types.voidptr,
        types.int32,
        output_state_type,
    )
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def init_table_builder_state(expected_state_type=None):
    """Initialize the C++ TableBuilderState pointer"""
    expected_state_type = unwrap_typeref(expected_state_type)
    if is_overload_none(expected_state_type):
        output_type = TableBuilderStateType()
    else:
        output_type = expected_state_type

    arr_dtypes = output_type.arr_ctypes
    arr_array_types = output_type.arr_array_types
    n_arrs = output_type.num_input_arrs

    def impl(
        expected_state_type=None,
    ):  # pragma: no cover
        return _init_table_builder_state(
            arr_dtypes.ctypes,
            arr_array_types.ctypes,
            n_arrs,
            output_type,
        )

    return impl


@intrinsic
def _table_builder_append(
    typingctx,
    builder_state,
    cpp_table,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="table_builder_append_py_entry"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    sig = types.void(builder_state, cpp_table)
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def table_builder_append(builder_state, table):
    """Append a table to the builder"""
    n_table_cols = builder_state.num_input_arrs
    in_col_inds = MetaType(tuple(range(n_table_cols)))

    def impl(builder_state, table):  # pragma: no cover
        cpp_table = py_data_to_cpp_table(table, (), in_col_inds, n_table_cols)
        _table_builder_append(builder_state, cpp_table)

    return impl


@intrinsic
def _table_builder_finalize(
    typingctx,
    builder_state,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [lir.IntType(8).as_pointer()],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="table_builder_finalize"
        )
        table_ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return table_ret

    ret_type = cpp_table_type
    sig = ret_type(builder_state)
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def table_builder_finalize(builder_state):
    """Finalize the builder and output a python table"""
    out_table_type = builder_state.build_table_type

    if out_table_type == types.unknown:
        out_cols_arr = np.array([], dtype=np.int64)
    else:
        num_cols = len(out_table_type.arr_types)
        out_cols_arr = np.array(range(num_cols), dtype=np.int64)

    def impl(
        builder_state,
    ):  # pragma: no cover
        out_cpp_table = _table_builder_finalize(builder_state)
        out_table = cpp_table_to_py_table(
            out_cpp_table, out_cols_arr, out_table_type, 0
        )
        delete_table(out_cpp_table)
        return out_table

    return impl
