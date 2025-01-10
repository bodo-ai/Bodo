"""Support for streaming sort (external sort) This file is mostly wrappers for
C++ implementations.
"""

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
from numba.extending import intrinsic, lower_builtin, models, register_model

import bodo
from bodo.ext import stream_sort_cpp
from bodo.libs.array import (
    cpp_table_to_py_table,
    delete_table,
    py_data_to_cpp_table,
)
from bodo.libs.array import table_type as cpp_table_type
from bodo.libs.streaming.base import StreamingStateType
from bodo.utils.transform import get_call_expr_arg
from bodo.utils.typing import (
    MetaType,
    get_overload_const_list,
    unwrap_typeref,
)

ll.add_symbol(
    "stream_sort_state_init_py_entry",
    stream_sort_cpp.stream_sort_state_init_py_entry,
)
ll.add_symbol(
    "stream_sort_build_consume_batch_py_entry",
    stream_sort_cpp.stream_sort_build_consume_batch_py_entry,
)
ll.add_symbol(
    "stream_sort_product_output_batch_py_entry",
    stream_sort_cpp.stream_sort_product_output_batch_py_entry,
)
ll.add_symbol(
    "delete_stream_sort_state",
    stream_sort_cpp.delete_stream_sort_state,
)


class SortStateType(StreamingStateType):
    """Type for C++ SortState pointer"""

    def __init__(self, build_table_type=types.unknown, key_indices=None):
        self._build_table_type = build_table_type
        self.key_indices = key_indices or []
        super().__init__(
            f"SortStateType(build_table={build_table_type}, key_indices={key_indices})"
        )

    def is_precise(self):
        return self._build_table_type != types.unknown

    def unify(self, typingctx, other):
        """Unify two GroupbyStateType instances when one doesn't have a resolved
        build_table_type.
        """
        if isinstance(other, SortStateType):
            if not other.is_precise() and self.is_precise():
                return self

            # Prefer the new type in case sort build changed its table type
            return other

    @cached_property
    def arr_dtypes(self) -> list[types.ArrayCompatible]:
        """Returns the list of types for each array in the build table."""
        if self.build_table_type == types.unknown:
            return []
        return self.build_table_type.arr_types

    @cached_property
    def mapped_arr_dtypes(self) -> list[types.ArrayCompatible]:
        """Returns the list of C++ types for each array in the build table."""
        if self.build_table_type == types.unknown:
            return []
        dtypes = self.arr_dtypes
        return [dtypes[i] for i in self.column_mapping]

    @cached_property
    def arr_ctypes(self) -> np.ndarray:
        return self._derive_c_types(self.mapped_arr_dtypes)

    @property
    def arr_array_types(self) -> np.ndarray:
        """
        Fetch the CArrayTypeEnum used for each array in the build table.

        Returns:
            List(int): The CArrayTypeEnum for each array in the build table. Note
                that C++ wants the actual integer but these are the values derived from
                CArrayTypeEnum.
        """
        return self._derive_c_array_types(self.mapped_arr_dtypes)

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
            return types.unknown
        return self._build_table_type

    @property
    def out_table_type(self):
        """The type of the output (same as input)"""
        return self.build_table_type

    @property
    def column_mapping(self):
        return list(self.key_indices) + [
            i for i in range(self.num_input_arrs) if i not in self.key_indices
        ]


register_model(SortStateType)(models.OpaqueModel)


@intrinsic
def _init_stream_sort_state(
    typingctx,
    output_state_type,
    operator_id,
    limit,
    offset,
    n_keys,
    vect_ascending,
    na_position,
    arr_ctypes,
    arr_array_types,
    n_arrs,
    parallel,
):
    output_type = unwrap_typeref(output_state_type)

    def codegen(context, builder, sig, args):
        (
            _,
            operator_id,
            limit,
            offset,
            n_keys,
            vect_ascending,
            na_position,
            arr_ctypes,
            arr_array_types,
            n_arrs,
            parallel,
        ) = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(1),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="stream_sort_state_init_py_entry"
        )
        ret = builder.call(
            fn_tp,
            (
                operator_id,
                limit,
                offset,
                n_keys,
                vect_ascending,
                na_position,
                arr_ctypes,
                arr_array_types,
                n_arrs,
                parallel,
            ),
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = output_type(
        output_state_type,
        types.int64,
        types.int64,
        types.int64,
        types.int64,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.int64,
        types.bool_,
    )
    return sig, codegen


def init_stream_sort_state(
    operator_id,
    limit,
    offset,
    by,
    asc_cols,
    na_position,
    col_names,
    parallel=False,
):
    pass


@infer_global(init_stream_sort_state)
class InitSortStateInfer(AbstractTemplate):
    """Typer for init_stream_sort_state that returns sort state type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(init_stream_sort_state)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        by = get_overload_const_list(folded_args[3])
        col_names = get_overload_const_list(folded_args[6])
        key_inds = tuple(col_names.index(col) for col in by)
        output_type = SortStateType(key_indices=key_inds)
        return signature(output_type, *folded_args).replace(pysig=pysig)


InitSortStateInfer._no_unliteral = True


@lower_builtin(init_stream_sort_state, types.VarArg(types.Any))
def lower_init_stream_sort_state(context, builder, sig, args):
    """lower init_stream_sort_state() using gen_init_stream_sort_state_impl"""
    impl = gen_init_stream_sort_state_impl(sig.return_type, *sig.args)
    return context.compile_internal(builder, impl, sig, args)


def gen_init_stream_sort_state_impl(
    output_type,
    operator_id,
    limit,
    offset,
    by,
    asc_cols,
    na_position,
    col_names,
    parallel=False,
):
    """Initialize the C++ TableBuilderState pointer"""

    asc_cols_ = get_overload_const_list(asc_cols)
    asc_cols_ = [int(asc) for asc in asc_cols_]
    na_position_ = get_overload_const_list(na_position)
    na_position_ = [int(pos == "last") for pos in na_position_]
    vect_asc = np.array(asc_cols_, np.int64)
    na_pos = np.array(na_position_, np.int64)

    arr_ctypes = output_type.arr_ctypes
    arr_array_types = output_type.arr_array_types
    n_arrs = len(arr_ctypes)

    def impl(
        operator_id,
        limit,
        offset,
        by,
        asc_cols,
        na_position,
        col_names,
        parallel=False,
    ):  # pragma: no cover
        return _init_stream_sort_state(
            output_type,
            operator_id,
            limit,
            offset,
            len(by),
            vect_asc.ctypes,
            na_pos.ctypes,
            arr_ctypes.ctypes,
            arr_array_types.ctypes,
            n_arrs,
            parallel,
        )

    return impl


@intrinsic
def _delete_stream_sort_state(
    typingctx,
    sort_state,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [lir.IntType(8).as_pointer()],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="delete_stream_sort_state"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    ret_type = types.void
    sig = ret_type(sort_state)
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def delete_stream_sort_state(sort_state):
    """Deletes the stream sort state."""

    def impl(
        sort_state,
    ):  # pragma: no cover
        _delete_stream_sort_state(sort_state)

    return impl


@intrinsic
def _sort_build_consume_batch(typingctx, sort_state, cpp_table, is_last):
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
            builder.module, fnty, name="stream_sort_build_consume_batch_py_entry"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.bool_(sort_state, cpp_table, is_last)
    return sig, codegen


def sort_build_consume_batch(sort_state, table, is_last):  # pragma: no cover
    pass


def gen_sort_build_consume_batch_impl(sort_state, table, is_last):
    """Consume a table batch in streaming sort

    Args:
        sort_state (SortState): C++ SortState pointer
        table (table_type): build table batch
        is_last (bool): is last batch locally
    Returns:
        bool: is last batch globally with possibility of false negatives due to iterations between syncs
    """
    n_table_cols = len(sort_state.build_table_type.arr_types)
    in_col_inds = MetaType(tuple(sort_state.column_mapping))

    def impl_sort_build_consume_batch(sort_state, table, is_last):  # pragma: no cover
        cpp_table = py_data_to_cpp_table(table, (), in_col_inds, n_table_cols)
        # TODO(aneesh) don't hardcode parallel as true
        return _sort_build_consume_batch(sort_state, cpp_table, is_last)

    return impl_sort_build_consume_batch


@infer_global(sort_build_consume_batch)
class SortBuildConsumeBatchInfer(AbstractTemplate):
    """Typer for sort_build_consume_batch that returns bool as output type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(sort_build_consume_batch)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        # Update state type in signature to include build table type from input
        state_type = folded_args[0]
        build_table_type = folded_args[1]
        new_state_type = SortStateType(build_table_type, state_type.key_indices)
        folded_args = (new_state_type, *folded_args[1:])
        return signature(types.bool_, *folded_args).replace(pysig=pysig)


@lower_builtin(sort_build_consume_batch, types.VarArg(types.Any))
def lower_sort_build_consume_batch(context, builder, sig, args):
    """lower sort_build_consume_batch() using gen_sort_build_consume_batch_impl above"""
    impl = gen_sort_build_consume_batch_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


@intrinsic
def _produce_output_batch(typingctx, sort_state, produce_output):
    def codegen(context, builder, sig, args):
        sort_state, produce_output = args
        out_is_last = cgutils.alloca_once(builder, lir.IntType(1))
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="stream_sort_product_output_batch_py_entry"
        )
        table_ret = builder.call(fn_tp, (sort_state, produce_output, out_is_last))
        items = [table_ret, builder.load(out_is_last)]
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return context.make_tuple(builder, sig.return_type, items)

    ret_type = types.Tuple([cpp_table_type, types.bool_])
    sig = ret_type(sort_state, produce_output)
    return sig, codegen


def produce_output_batch(sort_state, produce_output):  # pragma: no cover
    pass


def gen_produce_output_batch_impl(sort_state, produce_output):
    """Produce an output batch in streaming sort if one is available

    Args:
        sort_state (SortState): C++ SortState pointer
        produce_output (bool): should produce output
    Returns:
        bool: is last batch globally with possibility of false negatives due to iterations between syncs
    """
    # Undo the mapping in column_mapping (move key columns from prefix back to
    # their original locations)
    out_cols_arr = np.zeros(len(sort_state.column_mapping), dtype=np.int64)
    # sort_state.column_mapping[i] maps column i to some new location, so we
    # need to do the inverse and map the new location back to i.
    for i, c in enumerate(sort_state.column_mapping):
        out_cols_arr[c] = i
    out_table_type = sort_state.out_table_type

    def impl_produce_output_batch(sort_state, produce_output):  # pragma: no cover
        out_cpp_table, out_is_last = _produce_output_batch(sort_state, produce_output)
        out_table = cpp_table_to_py_table(
            out_cpp_table, out_cols_arr, out_table_type, 0
        )
        delete_table(out_cpp_table)
        return out_table, out_is_last

    return impl_produce_output_batch


@infer_global(produce_output_batch)
class SortFinalizeInfer(AbstractTemplate):
    """Typer for produce_output_batch that returns bool as output type"""

    def generic(self, args, kws):
        kws = dict(kws)
        sort_state = get_call_expr_arg(
            "produce_output_batch", args, kws, 0, "sort_state"
        )
        StreamingStateType.ensure_known_inputs(
            "produce_output_batch",
            (sort_state._build_table_type,),
        )

        # Output is (out_table, out_is_last)
        output_type = types.BaseTuple.from_types(
            (sort_state.out_table_type, types.bool_)
        )

        pysig = numba.core.utils.pysignature(produce_output_batch)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(output_type, *folded_args).replace(pysig=pysig)


@lower_builtin(produce_output_batch, types.VarArg(types.Any))
def lower_produce_output_batch(context, builder, sig, args):
    """lower produce_output_batch() using gen_produce_output_batch_impl above"""
    impl = gen_produce_output_batch_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)
