# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""Interface to C++ TableBuilderState/ChunkedTableBuilderState"""
from functools import cached_property
from typing import List

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
from numba.extending import (
    intrinsic,
    lower_builtin,
    models,
    overload,
    register_model,
)

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
    error_on_unsupported_nested_arrays,
    get_overload_const_bool,
    is_overload_none,
    unwrap_typeref,
)
from bodo.utils.utils import numba_to_c_array_types, numba_to_c_types

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
ll.add_symbol(
    "table_builder_get_data",
    table_builder_cpp.table_builder_get_data,
)
ll.add_symbol(
    "table_builder_reset",
    table_builder_cpp.table_builder_reset,
)
ll.add_symbol(
    "table_builder_nbytes_py_entry", table_builder_cpp.table_builder_nbytes_py_entry
)
ll.add_symbol(
    "delete_table_builder_state",
    table_builder_cpp.delete_table_builder_state,
)
ll.add_symbol(
    "chunked_table_builder_state_init_py_entry",
    table_builder_cpp.chunked_table_builder_state_init_py_entry,
)
ll.add_symbol(
    "chunked_table_builder_append_py_entry",
    table_builder_cpp.chunked_table_builder_append_py_entry,
)
ll.add_symbol(
    "chunked_table_builder_pop_chunk",
    table_builder_cpp.chunked_table_builder_pop_chunk,
)
ll.add_symbol(
    "delete_chunked_table_builder_state",
    table_builder_cpp.delete_chunked_table_builder_state,
)


class TableBuilderStateType(types.Type):
    """Type for C++ TableBuilderState pointer"""

    def __init__(
        self,
        build_table_type=types.unknown,
        is_chunked_builder=False,
    ):
        # TODO[BSE-937]: support nested arrays in streaming
        error_on_unsupported_nested_arrays(build_table_type)
        self._build_table_type = build_table_type
        self.is_chunked_builder = is_chunked_builder
        super().__init__(
            f"TableBuilderStateType(build_table={build_table_type}, is_chunked_builder={is_chunked_builder})"
        )

    @staticmethod
    def _derive_c_types(arr_types: List[types.ArrayCompatible]) -> np.ndarray:
        """Generate the CType Enum types for each array in the
        C++ build table via the indices.

        Args:
            arr_types (List[types.ArrayCompatible]): The array types to use.

        Returns:
            List(int): List with the integer values of each CTypeEnum value.
        """
        return numba_to_c_types(arr_types)

    @cached_property
    def arr_dtypes(self) -> List[types.ArrayCompatible]:
        """Returns the list of types for each array in the build table."""
        return self.build_table_type.arr_types

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
        return numba_to_c_array_types(arr_types)

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
        return len(self.arr_dtypes)

    @property
    def build_table_type(self):
        if self._build_table_type == types.unknown:
            return bodo.TableType(())
        else:
            return self._build_table_type


register_model(TableBuilderStateType)(models.OpaqueModel)


@intrinsic(prefer_literal=True)
def _init_table_builder_state(
    typingctx,
    arr_ctypes,
    arr_array_ctypes,
    n_arrs,
    output_state_type,
    input_dicts_unified,
):
    output_type = unwrap_typeref(output_state_type)

    def codegen(context, builder, sig, args):
        (
            arr_ctypes,
            arr_array_ctypes,
            n_arrs,
            _,
            in_dicts_unified,
        ) = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(1),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="table_builder_state_init_py_entry"
        )
        ret = builder.call(
            fn_tp, (arr_ctypes, arr_array_ctypes, n_arrs, in_dicts_unified)
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = output_type(
        types.voidptr,
        types.voidptr,
        types.int32,
        output_state_type,
        types.bool_,
    )
    return sig, codegen


@intrinsic(prefer_literal=True)
def _init_chunked_table_builder_state(
    typingctx, arr_ctypes, arr_array_ctypes, n_arrs, output_state_type, chunk_size
):
    output_type = unwrap_typeref(output_state_type)

    def codegen(context, builder, sig, args):
        (
            arr_ctypes,
            arr_array_ctypes,
            n_arrs,
            _,
            chunk_size,
        ) = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(64),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="chunked_table_builder_state_init_py_entry"
        )
        ret = builder.call(fn_tp, (arr_ctypes, arr_array_ctypes, n_arrs, chunk_size))
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = output_type(
        types.voidptr,
        types.voidptr,
        types.int32,
        output_state_type,
        chunk_size,
    )
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def init_table_builder_state(
    operator_id,
    expected_state_type=None,
    use_chunked_builder=False,
    input_dicts_unified=False,
):
    """Initialize the C++ TableBuilderState pointer"""
    expected_state_type = unwrap_typeref(expected_state_type)
    if is_overload_none(expected_state_type):
        output_type = TableBuilderStateType()
    else:
        output_type = expected_state_type

    arr_dtypes = output_type.arr_ctypes
    arr_array_types = output_type.arr_array_types
    n_arrs = output_type.num_input_arrs

    if not is_overload_none(expected_state_type) and get_overload_const_bool(
        use_chunked_builder
    ):
        assert (
            expected_state_type.is_chunked_builder
        ), "Error in init_table_builder_state: expected_state_type.is_chunked_builder must be True if use_chunked_builder is True"

        def impl(
            operator_id,
            expected_state_type=None,
            use_chunked_builder=False,
            input_dicts_unified=False,
        ):  # pragma: no cover
            return _init_chunked_table_builder_state(
                arr_dtypes.ctypes,
                arr_array_types.ctypes,
                n_arrs,
                output_type,
                bodo.bodosql_streaming_batch_size,
            )

    else:

        def impl(
            operator_id,
            expected_state_type=None,
            use_chunked_builder=False,
            input_dicts_unified=False,
        ):  # pragma: no cover
            return _init_table_builder_state(
                arr_dtypes.ctypes,
                arr_array_types.ctypes,
                n_arrs,
                output_type,
                input_dicts_unified,
            )

    return impl


@intrinsic(prefer_literal=True)
def _chunked_table_builder_append(
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
            builder.module, fnty, name="chunked_table_builder_append_py_entry"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    sig = types.void(builder_state, cpp_table)
    return sig, codegen


@intrinsic(prefer_literal=True)
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


@intrinsic(prefer_literal=True)
def _table_builder_nbytes(
    typingctx,
    builder_state,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(64),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="table_builder_nbytes_py_entry"
        )
        res = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return res

    sig = types.int64(builder_state)
    return sig, codegen


def table_builder_append(builder_state, table):
    pass


def gen_table_builder_append_impl(builder_state, table):
    """Append a table to the builder"""
    n_table_cols = builder_state.num_input_arrs
    in_col_inds = MetaType(tuple(range(n_table_cols)))

    if not is_overload_none(builder_state) and builder_state.is_chunked_builder:

        def impl(builder_state, table):  # pragma: no cover
            cpp_table = py_data_to_cpp_table(table, (), in_col_inds, n_table_cols)
            _chunked_table_builder_append(builder_state, cpp_table)

    else:

        def impl(builder_state, table):  # pragma: no cover
            cpp_table = py_data_to_cpp_table(table, (), in_col_inds, n_table_cols)
            _table_builder_append(builder_state, cpp_table)

    return impl


@infer_global(table_builder_append)
class TableBuilderAppendInfer(AbstractTemplate):
    """Typer for table_builder_append that returns none"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(table_builder_append)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(types.none, *folded_args).replace(pysig=pysig)


TableBuilderAppendInfer._no_unliteral = True


@lower_builtin(table_builder_append, types.VarArg(types.Any))
def lower_table_builder_append(context, builder, sig, args):
    """lower table_builder_append() using gen_table_builder_append_impl above"""
    impl = gen_table_builder_append_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def table_builder_nbytes(builder_state):
    pass


@overload(table_builder_nbytes)
def overload_table_builder_nbytes(builder_state):
    """Determine the number of current bytes inside the table
    of the given table builder. Currently only supported for
    the regular table builder
    (TODO: Support chunked table builder with spilling)"""

    def impl(builder_state):
        return _table_builder_nbytes(builder_state)

    return impl


@intrinsic(prefer_literal=True)
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
    """
    Finalize the builder and output a python table
    (Only implemented for non-chunked
    TODO(Keaton) implement this for chunked: https://bodo.atlassian.net/browse/BSE-977)
    """
    out_table_type = builder_state.build_table_type

    num_cols = len(out_table_type.arr_types)
    out_cols_arr = np.array(range(num_cols), dtype=np.int64)

    if not is_overload_none(builder_state) and builder_state.is_chunked_builder:
        raise RuntimeError("Chunked table builder finalize not implemented")
    else:

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


@intrinsic(prefer_literal=True)
def _chunked_table_builder_pop_chunk(
    typingctx,
    builder_state,
    produce_output,
    force_return,
):
    """
    Returns a tuple of a (possibly empty) chunk of data from the builder and a boolean indicating if the
    builder is empty.
    """

    def codegen(context, builder, sig, args):
        out_is_last = cgutils.alloca_once(builder, lir.IntType(1))
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),  # builder state
                lir.IntType(1),  # produce output
                lir.IntType(1),  # force return (Currently hard coded to True)
                lir.IntType(1).as_pointer(),  # bool* is_last_output_chunk,
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="chunked_table_builder_pop_chunk"
        )
        full_func_args = args + (out_is_last,)
        table_ret = builder.call(fn_tp, full_func_args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        builder.load(out_is_last)
        items = [
            table_ret,
            builder.load(out_is_last),
        ]
        return context.make_tuple(builder, sig.return_type, items)

    ret_type = types.Tuple([cpp_table_type, types.bool_])
    sig = ret_type(builder_state, produce_output, force_return)
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def table_builder_pop_chunk(builder_state, produce_output=True):
    """Return a chunk of data from the builder (Only implemented for chunked table builder)

    Returns a tuple of a (possibly empty) chunk of data from the builder and a boolean indicating if the
    returned chunk is the last chunk.
    Args:
    produce_output: If False, no data will be emitted from the builder, and this
                    function will return an empty table
    """
    out_table_type = builder_state.build_table_type

    num_cols = len(out_table_type.arr_types)
    out_cols_arr = np.array(range(num_cols), dtype=np.int64)

    if not is_overload_none(builder_state) and builder_state.is_chunked_builder:

        def impl(builder_state, produce_output=True):  # pragma: no cover
            out_cpp_table, is_last = _chunked_table_builder_pop_chunk(
                builder_state, produce_output, True
            )
            out_table = cpp_table_to_py_table(
                out_cpp_table, out_cols_arr, out_table_type, 0
            )
            delete_table(out_cpp_table)
            return out_table, is_last

    else:
        raise RuntimeError("Chunked table builder finalize not implemented")

    return impl


@intrinsic(prefer_literal=True)
def _delete_chunked_table_builder_state(
    typingctx,
    builder_state,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [lir.IntType(8).as_pointer()],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="delete_chunked_table_builder_state"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    ret_type = types.void
    sig = ret_type(builder_state)
    return sig, codegen


@intrinsic(prefer_literal=True)
def _delete_table_builder_state(
    typingctx,
    builder_state,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [lir.IntType(8).as_pointer()],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="delete_table_builder_state"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    ret_type = types.void
    sig = ret_type(builder_state)
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def delete_table_builder_state(builder_state):
    """Deletes the table builder state."""

    if not is_overload_none(builder_state) and builder_state.is_chunked_builder:

        def impl(
            builder_state,
        ):  # pragma: no cover
            _delete_chunked_table_builder_state(builder_state)

    else:

        def impl(
            builder_state,
        ):  # pragma: no cover
            _delete_table_builder_state(builder_state)

    return impl


@intrinsic(prefer_literal=True)
def _table_builder_get_data(
    typingctx,
    builder_state,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [lir.IntType(8).as_pointer()],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="table_builder_get_data"
        )
        table_ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return table_ret

    ret_type = cpp_table_type
    sig = ret_type(builder_state)
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def table_builder_get_data(builder_state):
    """Get builder data as a Python table without finalizing or affecting state"""
    out_table_type = builder_state.build_table_type

    num_cols = len(out_table_type.arr_types)
    out_cols_arr = np.array(range(num_cols), dtype=np.int64)

    def impl(
        builder_state,
    ):  # pragma: no cover
        out_cpp_table = _table_builder_get_data(builder_state)
        out_table = cpp_table_to_py_table(
            out_cpp_table, out_cols_arr, out_table_type, 0
        )
        delete_table(out_cpp_table)
        return out_table

    return impl


@intrinsic(prefer_literal=True)
def table_builder_reset(
    typingctx,
    builder_state,
):
    """Reset table builder's buffer (sets array buffer sizes to zero but keeps capacity the same)"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [lir.IntType(8).as_pointer()],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="table_builder_reset"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return

    sig = types.none(builder_state)
    return sig, codegen
