from textwrap import dedent
from typing import TYPE_CHECKING, List

import llvmlite.binding as ll
import numba
import numpy as np
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.ir_utils import next_label
from numba.extending import intrinsic, models, register_model

from bodo.hiframes.pd_dataframe_ext import init_dataframe
from bodo.hiframes.pd_index_ext import init_range_index
from bodo.hiframes.table import TableType, set_table_len
from bodo.io import arrow_cpp
from bodo.io.helpers import map_cpp_to_py_table_column_idxs
from bodo.libs.array import cpp_table_to_py_table, delete_table, table_type
from bodo.utils.typing import BodoError, is_overload_none
from bodo.utils.utils import MetaType, inlined_check_and_propagate_cpp_exception

if TYPE_CHECKING:  # pragma: no cover
    from llvmlite.ir.builder import IRBuilder
    from numba.core.base import BaseContext


ll.add_symbol("arrow_reader_read_py_entry", arrow_cpp.arrow_reader_read_py_entry)
ll.add_symbol("arrow_reader_del_py_entry", arrow_cpp.arrow_reader_del_py_entry)


class ArrowReaderType(types.Type):
    def __init__(
        self, col_names: List[str], col_types: List[types.ArrayCompatible]
    ):  # pragma: no cover
        self.col_names = col_names
        self.col_types = col_types
        super().__init__(f"ArrowReaderMetaType({col_names}, {col_types})")


register_model(ArrowReaderType)(models.OpaqueModel)


@intrinsic
def arrow_reader_read_py_entry(typingctx, arrow_reader_t):  # pragma: no cover
    """
    Get the next batch from a C++ ArrowReader object
    """
    assert isinstance(arrow_reader_t, ArrowReaderType)
    ret_type = types.Tuple([table_type, types.boolean, types.int64])

    def codegen(context: "BaseContext", builder: "IRBuilder", signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),  # void*
            [
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(1).as_pointer(),  # bool*
                lir.IntType(64).as_pointer(),  # uint64*
            ],
        )

        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="arrow_reader_read_py_entry"
        )

        # Allocate values to point to
        is_last_out_ptr = cgutils.alloca_once(builder, lir.IntType(1))
        num_rows_ptr = cgutils.alloca_once(builder, lir.IntType(64))
        total_args = args + (is_last_out_ptr, num_rows_ptr)
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

    sig = ret_type(arrow_reader_t)
    return sig, codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def read_arrow_next(arrow_reader, used_cols=None):  # pragma: no cover
    if not isinstance(arrow_reader, ArrowReaderType):
        raise BodoError(
            "read_arrow_next(): First argument arrow_reader must be an ArrowReader"
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

    call_id = next_label()
    table_idx_var = f"table_idx_{call_id}"
    py_table_type_var = f"py_table_type_{call_id}"

    func_text = dedent(
        f"""\
    def func(arrow_reader, used_cols=None):
        out_table, is_last_out, num_rows = arrow_reader_read_py_entry(arrow_reader)
        table_var = cpp_table_to_py_table(out_table, {table_idx_var}, {py_table_type_var}, num_rows)
        delete_table(out_table)
        table_var = set_table_len(table_var, num_rows)
        return table_var, is_last_out
    """
    )

    glbls = {
        "arrow_reader_read_py_entry": arrow_reader_read_py_entry,
        "init_range_index": init_range_index,
        "init_dataframe": init_dataframe,
        "cpp_table_to_py_table": cpp_table_to_py_table,
        "set_table_len": set_table_len,
        "delete_table": delete_table,
        table_idx_var: used_col_values,
        py_table_type_var: TableType(tuple(arrow_reader.col_types)),
    }

    l = {}
    exec(func_text, glbls, l)
    return l["func"]


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
