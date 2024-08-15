from typing import TYPE_CHECKING

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
from bodo.hiframes.table import TableType, set_table_len
from bodo.io import arrow_cpp
from bodo.io.helpers import map_cpp_to_py_table_column_idxs
from bodo.libs.array import cpp_table_to_py_table, delete_table, table_type
from bodo.utils.transform import get_call_expr_arg
from bodo.utils.typing import BodoError, is_overload_none
from bodo.utils.utils import MetaType, inlined_check_and_propagate_cpp_exception

if TYPE_CHECKING:  # pragma: no cover
    from llvmlite.ir.builder import IRBuilder
    from numba.core.base import BaseContext


ll.add_symbol("arrow_reader_read_py_entry", arrow_cpp.arrow_reader_read_py_entry)
ll.add_symbol("arrow_reader_del_py_entry", arrow_cpp.arrow_reader_del_py_entry)


class ArrowReaderType(types.Type):
    def __init__(
        self, col_names: list[str], col_types: list[types.ArrayCompatible]
    ):  # pragma: no cover
        self.col_names = col_names
        self.col_types = col_types
        super().__init__(f"ArrowReaderMetaType({col_names}, {col_types})")


register_model(ArrowReaderType)(models.OpaqueModel)


@intrinsic
def arrow_reader_read_py_entry(
    typingctx, arrow_reader_t, produce_output
):  # pragma: no cover
    """
    Get the next batch from a C++ ArrowReader object
    """
    assert isinstance(arrow_reader_t, ArrowReaderType)
    assert isinstance(produce_output, numba.types.Boolean)
    ret_type = types.Tuple([table_type, types.boolean, types.int64])

    def codegen(context: "BaseContext", builder: "IRBuilder", signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),  # void*
            [
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(1).as_pointer(),  # bool*
                lir.IntType(64).as_pointer(),  # uint64*
                lir.IntType(1),  # bool
            ],
        )

        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="arrow_reader_read_py_entry"
        )

        # Allocate values to point to
        is_last_out_ptr = cgutils.alloca_once(builder, lir.IntType(1))
        num_rows_ptr = cgutils.alloca_once(builder, lir.IntType(64))
        total_args = (args[0], is_last_out_ptr, num_rows_ptr, args[1])
        table = builder.call(fn_tp, total_args)
        inlined_check_and_propagate_cpp_exception(context, builder)

        # Fetch the underlying data from the pointers.
        items = [
            table,
            builder.load(is_last_out_ptr),
            builder.load(num_rows_ptr),
        ]
        # Return the tuple
        return context.make_tuple(builder, ret_type, items)

    sig = ret_type(arrow_reader_t, produce_output)
    return sig, codegen


def read_arrow_next(arrow_reader, produce_output, used_cols=None):  # pragma: no cover
    pass


def gen_read_arrow_next_impl(arrow_reader, produce_output, used_cols=None):
    if not isinstance(arrow_reader, ArrowReaderType):  # pragma: no cover
        raise BodoError(
            f"read_arrow_next(): First argument arrow_reader must be an ArrowReader type, not {arrow_reader}"
        )

    if is_overload_none(used_cols):
        used_col_values = np.arange(len(arrow_reader.col_names), dtype=np.int64)
    else:
        assert isinstance(used_cols, types.TypeRef) and isinstance(
            used_cols.instance_type, MetaType
        )
        used_col_values = map_cpp_to_py_table_column_idxs(
            arrow_reader.col_names, used_cols.instance_type.meta
        )

    table_type = TableType(tuple(arrow_reader.col_types))

    def impl_read_arrow_next(
        arrow_reader, produce_output, used_cols=None
    ):  # pragma: no cover
        out_table, is_last_out, num_rows = arrow_reader_read_py_entry(
            arrow_reader, produce_output
        )
        table_var = cpp_table_to_py_table(
            out_table, used_col_values, table_type, num_rows
        )
        delete_table(out_table)
        table_var = set_table_len(table_var, num_rows)
        return table_var, is_last_out

    return impl_read_arrow_next


@infer_global(read_arrow_next)
class ReadArrowNextInfer(AbstractTemplate):
    """Typer for read_arrow_next that returns (output_table_type, bool)
    as output type.
    """

    def generic(self, args, kws):
        kws = dict(kws)
        arrow_reader = get_call_expr_arg(
            "read_arrow_next", args, kws, 0, "arrow_reader"
        )
        out_table_type = TableType(tuple(arrow_reader.col_types))
        # Output is (out_table, out_is_last)
        output_type = types.BaseTuple.from_types((out_table_type, types.bool_))

        pysig = numba.core.utils.pysignature(read_arrow_next)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(output_type, *folded_args).replace(pysig=pysig)


ReadArrowNextInfer._no_unliteral = True


@lower_builtin(read_arrow_next, types.VarArg(types.Any))
def lower_read_arrow_next(context, builder, sig, args):
    """lower read_arrow_next() using gen_read_arrow_next_impl above"""
    impl = gen_read_arrow_next_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


@intrinsic
def arrow_reader_del(typingctx, arrow_reader_t):  # pragma: no cover
    """
    Delete an ArrowReader object by calling the `delete` keyword in C++
    """
    assert isinstance(arrow_reader_t, ArrowReaderType)

    def codegen(context: "BaseContext", builder: "IRBuilder", signature, args):
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])  # void*

        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="arrow_reader_del_py_entry"
        )
        builder.call(fn_tp, args)
        inlined_check_and_propagate_cpp_exception(context, builder)
        return

    sig = types.void(arrow_reader_t)
    return sig, codegen
