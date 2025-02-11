"""Interface to C++ TableBuilderState/ChunkedTableBuilderState"""

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
from bodo.libs.streaming.base import StreamingStateType
from bodo.utils.typing import (
    MetaType,
    error_on_unsupported_streaming_arrays,
    get_overload_const_bool,
    is_overload_none,
    unwrap_typeref,
)

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


class TableBuilderStateType(StreamingStateType):
    """Type for C++ TableBuilderState pointer"""

    def __init__(
        self,
        build_table_type=types.unknown,
        is_chunked_builder=False,
    ):
        error_on_unsupported_streaming_arrays(build_table_type)
        self._build_table_type = build_table_type
        self.is_chunked_builder = is_chunked_builder
        super().__init__(
            f"TableBuilderStateType(build_table={build_table_type}, is_chunked_builder={is_chunked_builder})"
        )

    def is_precise(self):
        return self._build_table_type != types.unknown

    def unify(self, typingctx, other):
        """Unify two TableBuilderStateType instances when one doesn't have a resolved
        build_table_type.
        """
        if (
            isinstance(other, TableBuilderStateType)
            and self.is_chunked_builder == other.is_chunked_builder
        ):
            if not other.is_precise() and self.is_precise():
                return self
            # Prefer the new type in case append changed its table type
            return other

    @cached_property
    def arr_dtypes(self) -> list[types.ArrayCompatible]:
        """Returns the list of types for each array in the build table."""
        return self.build_table_type.arr_types

    @cached_property
    def arr_ctypes(self) -> np.ndarray:
        return self._derive_c_types(self.arr_dtypes)

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


@intrinsic
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


@intrinsic
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


def init_table_builder_state(
    operator_id,
    expected_state_type=None,
    use_chunked_builder=False,
    input_dicts_unified=False,
):
    pass


@infer_global(init_table_builder_state)
class InitTableBuilderStateInfer(AbstractTemplate):
    """Typer for init_table_builder_state that returns table builder state type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(init_table_builder_state)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        expected_state_type = unwrap_typeref(folded_args[1])
        is_chunked_builder = get_overload_const_bool(folded_args[2])
        if is_overload_none(expected_state_type):
            output_type = TableBuilderStateType(is_chunked_builder=is_chunked_builder)
        else:
            output_type = expected_state_type

        return signature(output_type, *folded_args).replace(pysig=pysig)


InitTableBuilderStateInfer._no_unliteral = True


@lower_builtin(init_table_builder_state, types.VarArg(types.Any))
def lower_init_table_builder_state(context, builder, sig, args):
    """lower init_table_builder_state() using gen_init_table_builder_state_impl"""
    impl = gen_init_table_builder_state_impl(sig.return_type, *sig.args)
    return context.compile_internal(builder, impl, sig, args)


def gen_init_table_builder_state_impl(
    output_type,
    operator_id,
    expected_state_type=None,
    use_chunked_builder=False,
    input_dicts_unified=False,
):
    """Initialize the C++ TableBuilderState pointer"""

    arr_dtypes = output_type.arr_ctypes
    arr_array_types = output_type.arr_array_types

    # We can just pass the length of the serialized types directly, since on the C++ side we immediately deserialize.
    n_arrs = len(arr_array_types)

    if get_overload_const_bool(use_chunked_builder):
        assert output_type.is_chunked_builder, (
            "Error in init_table_builder_state: expected_state_type.is_chunked_builder must be True if use_chunked_builder is True"
        )

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


@intrinsic
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


@intrinsic
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

    if builder_state.is_chunked_builder:

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
        # Update state type in signature to include build table type from input
        state_type = folded_args[0]
        build_table_type = folded_args[1]
        new_state_type = TableBuilderStateType(
            build_table_type, state_type.is_chunked_builder
        )
        folded_args = (new_state_type, *folded_args[1:])
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


def table_builder_finalize(builder_state):
    pass


@infer_global(table_builder_finalize)
class TableBuilderFinalizeInfer(AbstractTemplate):
    """Typer for table_builder_finalize that returns table type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(table_builder_finalize)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        builder_state = unwrap_typeref(folded_args[0])
        StreamingStateType.ensure_known_inputs(
            "table_builder_finalize",
            (builder_state._build_table_type,),
        )
        return signature(builder_state.build_table_type, *folded_args).replace(
            pysig=pysig
        )


TableBuilderFinalizeInfer._no_unliteral = True


@lower_builtin(table_builder_finalize, types.VarArg(types.Any))
def lower_table_builder_finalize(context, builder, sig, args):
    """lower table_builder_finalize() using gen_table_builder_finalize_impl"""
    impl = gen_table_builder_finalize_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def gen_table_builder_finalize_impl(builder_state):
    """
    Finalize the builder and output a python table
    (Only implemented for non-chunked
    TODO(Keaton) implement this for chunked: https://bodo.atlassian.net/browse/BSE-977)
    """
    out_table_type = builder_state.build_table_type

    num_cols = len(out_table_type.arr_types)
    out_cols_arr = np.array(range(num_cols), dtype=np.int64)

    if builder_state.is_chunked_builder:
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


@intrinsic
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


def table_builder_pop_chunk(builder_state, produce_output=True):
    pass


@infer_global(table_builder_pop_chunk)
class TableBuilderPopChunkInfer(AbstractTemplate):
    """Typer for table_builder_pop_chunk that returns table type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(table_builder_pop_chunk)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        builder_state = folded_args[0]
        StreamingStateType.ensure_known_inputs(
            "table_builder_pop_chunk",
            (builder_state._build_table_type,),
        )
        output_type = types.BaseTuple.from_types(
            (builder_state.build_table_type, types.bool_)
        )
        return signature(output_type, *folded_args).replace(pysig=pysig)


TableBuilderPopChunkInfer._no_unliteral = True


@lower_builtin(table_builder_pop_chunk, types.VarArg(types.Any))
def lower_table_builder_pop_chunk(context, builder, sig, args):
    """lower table_builder_pop_chunk() using gen_table_builder_pop_chunk_impl below"""
    impl = gen_table_builder_pop_chunk_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def gen_table_builder_pop_chunk_impl(builder_state, produce_output=True):
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

    if builder_state.is_chunked_builder:

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
        raise RuntimeError("TableBuildBuffer finalize not implemented")

    return impl


@intrinsic
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


@intrinsic
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

    if builder_state.is_chunked_builder:

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


@intrinsic
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


def table_builder_get_data(builder_state):
    pass


@infer_global(table_builder_get_data)
class TableBuilderGetDataInfer(AbstractTemplate):
    """Typer for table_builder_get_data that returns table type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(table_builder_get_data)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        builder_state = folded_args[0]
        StreamingStateType.ensure_known_inputs(
            "table_builder_get_data",
            (builder_state._build_table_type,),
        )
        output_type = builder_state.build_table_type
        return signature(output_type, *folded_args).replace(pysig=pysig)


TableBuilderGetDataInfer._no_unliteral = True


@lower_builtin(table_builder_get_data, types.VarArg(types.Any))
def lower_table_builder_get_data(context, builder, sig, args):
    """lower table_builder_get_data() using gen_table_builder_get_data_impl below"""
    impl = gen_table_builder_get_data_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def gen_table_builder_get_data_impl(builder_state):
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


@intrinsic
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
