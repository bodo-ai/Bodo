# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Support for streaming union.
"""
from functools import cached_property
from typing import Optional, Tuple

import numba
import numpy as np
from numba.core import types
from numba.core.typing.templates import (
    AbstractTemplate,
    infer_global,
    signature,
)
from numba.extending import lower_builtin, models, register_model

import bodo
from bodo.hiframes.table import TableType
from bodo.libs.array import (
    cpp_table_to_py_table,
    delete_table,
    py_data_to_cpp_table,
)
from bodo.utils.transform import get_call_expr_arg
from bodo.utils.typing import (
    BodoError,
    MetaType,
    dtype_to_array_type,
    error_on_unsupported_nested_arrays,
    get_common_scalar_dtype,
    get_overload_const_bool,
    is_nullable_ignore_sentinals,
    is_overload_bool,
    is_overload_none,
    unwrap_typeref,
)


class UnionStateType(types.Type):
    all: bool
    in_table_types: Tuple[TableType, ...]

    def __init__(
        self,
        all: bool = False,
        in_table_types: Tuple[TableType, ...] = (),
    ):
        # TODO[BSE-937]: support nested arrays in streaming
        for in_table_type in in_table_types:
            error_on_unsupported_nested_arrays(in_table_type)

        self.all = all
        self.in_table_types = in_table_types
        super().__init__(f"UnionStateType(all={all}, in_table_types={in_table_types})")

    @property
    def key(self):
        return (self.all, self.in_table_types)

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
        total_idxs.append(np.int32(10))

        for key_idx in key_indices:
            total_idxs.append(key_idx)

        idx_set = set(key_indices)
        for i in range(num_cols):
            if i not in idx_set:
                total_idxs.append(i)
        return tuple(total_idxs)

    @cached_property
    def n_cols(self) -> int:
        if len(self.in_table_types) == 0:
            return 0
        return len(self.in_table_types[0].arr_types)

    @property
    def n_keys(self) -> int:
        """
        Number of keys in UNION DISTINCT case
        Intended for GroupBy Compatibility, otherwise use n_cols
        """
        return self.n_cols

    @cached_property
    def out_table_type(self):
        if len(self.in_table_types) == 0:
            return types.unknown

        num_cols = len(self.in_table_types[0].arr_types)
        for in_table_type in self.in_table_types:
            if not isinstance(in_table_type, TableType):
                raise BodoError("stream_union.py: Must be called with tables")
            if num_cols != len(in_table_type.arr_types):
                raise BodoError(
                    "stream_union.py: Must be called with tables with the same number of columns"
                )

        if len(self.in_table_types) == 1:
            return self.in_table_types[0]

        # TODO: Refactor common code between non-streaming union
        # and streaming join for key columns
        out_arr_types = []
        for i in range(num_cols):
            in_col_types = [
                in_table_type.arr_types[i] for in_table_type in self.in_table_types
            ]
            is_nullable_out_col = any(
                col_type == bodo.null_array_type
                or is_nullable_ignore_sentinals(col_type)
                for col_type in in_col_types
            )

            if len(in_col_types) == 0:
                out_arr_types.append(bodo.null_array_type)

            elif all(in_col_types[0] == col_typ for col_typ in in_col_types):
                out_arr_types.append(in_col_types[0])

            elif any(col_typ == bodo.dict_str_arr_type for col_typ in in_col_types):
                for col_type in in_col_types:
                    if col_type not in (
                        bodo.dict_str_arr_type,
                        bodo.string_array_type,
                        bodo.null_array_type,
                    ):
                        raise BodoError(
                            f"Unable to union table with columns of incompatible types {col_type} and {bodo.dict_str_arr_type} in column {i}."
                        )
                out_arr_types.append(bodo.dict_str_arr_type)

            else:
                dtype, _ = get_common_scalar_dtype(
                    [t.dtype for t in in_col_types], allow_downcast=True
                )
                if dtype is None:
                    raise BodoError(
                        f"Unable to union table with columns of incompatible types. Found types {in_col_types} in column {i}."
                    )

                out_arr_types.append(dtype_to_array_type(dtype, is_nullable_out_col))

        return TableType(tuple(out_arr_types))


register_model(UnionStateType)(models.OpaqueModel)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def init_union_state(
    operator_id,
    all=False,
    expected_state_typeref=None,
    parallel=False,
):
    expected_state_type: Optional[UnionStateType] = unwrap_typeref(expected_state_typeref)  # type: ignore

    all_const = get_overload_const_bool(all)

    if is_overload_none(expected_state_type):
        output_type = UnionStateType(all=all_const)
    else:
        output_type: UnionStateType = expected_state_type  # type: ignore
        assert output_type.all == all_const

    arr_dtypes = np.array(
        []
        if output_type.out_table_type == types.unknown
        else output_type.out_table_type.c_dtypes,
        dtype=np.int8,
    )
    arr_array_types = np.array(
        []
        if output_type.out_table_type == types.unknown
        else output_type.out_table_type.c_array_types,
        dtype=np.int8,
    )
    n_arrs = output_type.n_cols

    if all_const:

        def impl(
            operator_id,
            all=False,
            expected_state_typeref=None,
            parallel=False,
        ):  # pragma: no cover
            return bodo.libs.table_builder._init_chunked_table_builder_state(
                arr_dtypes.ctypes,
                arr_array_types.ctypes,
                n_arrs,
                output_type,
                bodo.bodosql_streaming_batch_size,
            )

    else:
        # Distinct Only. No aggregation functions
        ftypes_arr = np.array([], np.int32)
        f_in_offsets = np.array([1], np.int32)
        f_in_cols = np.array([], np.int32)

        mrnf_sort_asc = np.array([], dtype=np.bool_)
        mrnf_sort_na = np.array([], dtype=np.bool_)
        mrnf_part_cols_to_keep = np.array([], dtype=np.bool_)
        mrnf_sort_cols_to_keep = np.array([], dtype=np.bool_)
        mrnf_n_sort_keys = 0

        def impl(
            operator_id,
            all=False,
            expected_state_typeref=None,
            parallel=False,
        ):
            return bodo.libs.stream_groupby._init_groupby_state(
                operator_id,
                arr_dtypes.ctypes,
                arr_array_types.ctypes,
                n_arrs,
                ftypes_arr.ctypes,
                f_in_offsets.ctypes,
                f_in_cols.ctypes,
                0,
                mrnf_sort_asc.ctypes,
                mrnf_sort_na.ctypes,
                mrnf_n_sort_keys,
                mrnf_part_cols_to_keep.ctypes,
                mrnf_sort_cols_to_keep.ctypes,
                -1,  # op_pool_size_bytes
                output_type,
                parallel,
            )

    return impl


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def _union_cast_batch(union_state: UnionStateType, table: TableType):
    """
    Internal function to cast table before UNION operation

    Args:
        union_state (UnionState): Union State. For this function,
            only used for casting info
        table (table_type): Input table batch
    Returns:
        table_type: Casted table argument
    """

    if (
        union_state.out_table_type == table
        or union_state.out_table_type == types.unknown
    ):
        return lambda union_state, table: table

    py_table_typ: TableType = union_state.out_table_type  # type: ignore

    def impl(union_state, table):  # pragma: no cover
        return bodo.utils.table_utils.table_astype(  # type: ignore
            table, py_table_typ, False, _bodo_nan_to_str=False
        )

    return impl


def union_consume_batch(union_state, table, is_last, is_final_pipeline):
    pass


def gen_union_consume_batch_impl(union_state, table, is_last, is_final_pipeline):
    """
    Consume a table batch in streaming union. Will cast the table
    and then process depending on type of union.

    Args:
        union_state (UnionState): Union State, containing internal
            state tool (Chunked Table Builder or Aggregation)
        table (table_type): Input table batch
        is_last (bool): is last batch (in this pipeline) locally
        is_final_pipeline (bool): Is this the final pipeline. Only relevant for the
         Union-Distinct case where this is called in multiple pipelines. For regular
         groupby, this should always be true.
    """

    if not isinstance(union_state, UnionStateType):  # pragma: no cover
        raise BodoError(
            f"union_cast_batch: Expected type UnionStateType "
            f"for first arg `union_state`, found {union_state}"
        )
    if not isinstance(table, TableType):  # pragma: no cover
        raise BodoError(
            f"union_cast_batch: Expected type TableType "
            f"for second arg `table`, found {table}"
        )

    n_cols = union_state.n_cols
    in_col_inds = MetaType(tuple(range(n_cols)))

    if union_state.all:

        def impl(union_state, table, is_last, is_final_pipeline):  # pragma: no cover
            casted_table = _union_cast_batch(union_state, table)
            cpp_table = py_data_to_cpp_table(casted_table, (), in_col_inds, n_cols)
            bodo.libs.table_builder._chunked_table_builder_append(
                union_state, cpp_table
            )
            return is_last

    else:

        def impl(union_state, table, is_last, is_final_pipeline):  # pragma: no cover
            casted_table = _union_cast_batch(union_state, table)
            cpp_table = py_data_to_cpp_table(casted_table, (), in_col_inds, n_cols)
            return bodo.libs.stream_groupby._groupby_build_consume_batch(
                union_state, cpp_table, is_last, is_final_pipeline
            )

    return impl


@infer_global(union_consume_batch)
class UnionConsumeBatchInfer(AbstractTemplate):
    """Typer for union_consume_batch that returns bool as output type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(union_consume_batch)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(types.bool_, *folded_args).replace(pysig=pysig)


UnionConsumeBatchInfer._no_unliteral = True


@lower_builtin(union_consume_batch, types.VarArg(types.Any))
def lower_union_consume_batch(context, builder, sig, args):
    """lower union_consume_batch() using gen_union_consume_batch_impl above"""
    impl = gen_union_consume_batch_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def union_produce_batch(union_state, produce_output=True):
    pass


def gen_union_produce_batch_impl(union_state, produce_output=True):
    """
    Return a chunk of data from UNION internal state

    Args:
        union_state (UnionState): Union State, containing internal
            state tool (Chunked Table Builder or Aggregation)
        produce_output (bool): If False, no data will be emitted
            from the builder, and this function will return an
            empty table

    Returns:
        table_type: Output table batch
        is_last: Returned last batch
    """

    if not isinstance(union_state, UnionStateType):  # pragma: no cover
        raise BodoError(
            f"union_produce_batch: Expected type UnionStateType "
            f"for first arg `union_state`, found {union_state}"
        )
    if not is_overload_bool(produce_output):  # pragma: no cover
        raise BodoError(
            f"union_produce_batch: Expected type bool "
            f"for second arg `produce_output`, found {produce_output}"
        )

    out_table_type = union_state.out_table_type
    out_cols_arr = np.array(range(union_state.n_cols), dtype=np.int64)

    if union_state.all:

        def impl(
            union_state, produce_output=True
        ) -> Tuple[TableType, bool]:  # pragma: no cover
            (
                out_cpp_table,
                is_last,
            ) = bodo.libs.table_builder._chunked_table_builder_pop_chunk(
                union_state, produce_output, True
            )
            out_table = cpp_table_to_py_table(
                out_cpp_table, out_cols_arr, out_table_type, 0
            )
            delete_table(out_cpp_table)
            return out_table, is_last

    else:

        def impl(
            union_state,
            produce_output=True,
        ) -> Tuple[TableType, bool]:  # pragma: no cover
            (
                out_cpp_table,
                out_is_last,
            ) = bodo.libs.stream_groupby._groupby_produce_output_batch(
                union_state, produce_output
            )
            out_table = cpp_table_to_py_table(
                out_cpp_table, out_cols_arr, out_table_type, 0
            )
            delete_table(out_cpp_table)
            return out_table, out_is_last

    return impl


@infer_global(union_produce_batch)
class UnionProduceOutputInfer(AbstractTemplate):
    """Typer for union_produce_batch that returns (output_table_type, bool)
    as output type.
    """

    def generic(self, args, kws):
        kws = dict(kws)
        union_state = get_call_expr_arg(
            "union_produce_batch", args, kws, 0, "union_state"
        )
        out_table_type = union_state.out_table_type
        # Output is (out_table, out_is_last)
        output_type = types.BaseTuple.from_types((out_table_type, types.bool_))

        pysig = numba.core.utils.pysignature(union_produce_batch)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(output_type, *folded_args).replace(pysig=pysig)


UnionProduceOutputInfer._no_unliteral = True


@lower_builtin(union_produce_batch, types.VarArg(types.Any))
def lower_union_produce_batch(context, builder, sig, args):
    """lower union_produce_batch() using gen_union_produce_batch_impl above"""
    impl = gen_union_produce_batch_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def delete_union_state(union_state):
    """
    Delete Union state runtime object
    """

    if not isinstance(union_state, UnionStateType):  # pragma: no cover
        raise BodoError(
            f"delete_union_state: Expected type UnionStateType "
            f"for first arg `union_state`, found {union_state}"
        )

    if union_state.all:
        return lambda union_state: bodo.libs.table_builder._delete_chunked_table_builder_state(
            union_state
        )
    else:
        return lambda union_state: bodo.libs.stream_groupby.delete_groupby_state(
            union_state
        )
