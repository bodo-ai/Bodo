# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements array kernels such as median and quantile.
"""

import llvmlite.binding as ll
import numpy as np
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic, overload

import bodo
from bodo.hiframes.table import TableType
from bodo.libs import lateral
from bodo.libs.array import (
    cpp_table_to_py_data,
    delete_table,
    py_data_to_cpp_table,
    table_type,
)
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.map_arr_ext import MapArrayType
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.utils.typing import (
    MetaType,
    get_overload_const_bool,
    get_overload_const_int,
    raise_bodo_error,
    unwrap_typeref,
)

ll.add_symbol("lateral_flatten", lateral.lateral_flatten_py_entrypt)

from bodo.libs.bodosql_array_kernels import get_combined_type


@intrinsic
def _lateral_flatten(
    typingctx,
    table_t,
    total_rows_t,
    output_seq_t,
    output_key_t,
    output_path_t,
    output_idx_t,
    output_val_t,
    output_this_t,
    json_mode_t,
    outer_t,
):
    assert table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),  # table_info*
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="lateral_flatten"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        table_type(
            table_t,
            types.voidptr,
            types.bool_,
            types.bool_,
            types.bool_,
            types.bool_,
            types.bool_,
            types.bool_,
            types.bool_,
            types.bool_,
        ),
        codegen,
    )


def lateral_flatten(
    in_table, keep_cols, explode_col, outputs, outer
):  # pragma: no cover
    # Dummy function to overload
    pass


@overload(lateral_flatten, no_unliteral=True)
def overload_lateral_flatten(in_table, keep_cols, explode_col, outputs, outer):
    """
    Kernel used to implement the SQL functionality LATERAL FLATTEN(A).
    Explodes the rows of column A from nested arrays into a column of the inner
    elements. Currently only supported when A is a column of strings.

    Args:
        in_table (TableType): the table to have its rows exploded.
        keep_cols (MetaType): which columns should be preserved?
        explode_col (integer): which column is the array column to be exploded?
        outputs (MetaType): which of the 6 output columns to include, as a boolean tuple:
            0: SEQ (not currently supported)
            1: KEY (not currently supported)
            2: PATH (not currently supported)
            3: INDEX
            4: VALUE
            5: THIS (supported unless the explode column is an array of strings)
        outer (boolean): if true, ensure 1 row is generated even when the
        explode col is null/empty.

    Returns:
        (TableType): the input table with the rows of the kept columns duplicated
        according to the number of entries in the array from the same row. The
        returned table has all of the output columns (in the
        same order as they are specified by the outputs argument) first followed by
        the replicated columns (in the same order as they are in the keep_cols tuple).

        For example, if keep_cols = (2, 5, 3, 6), explode_col=4, and
        outputs = (False, False, False, True, True, False), then the output
        will have the index column, followed by the value column, followed by
        columns 2/5/3/6 with their rows replicated.
    """
    assert isinstance(in_table, TableType)
    in_arr_types = in_table.arr_types

    # Find the column that is to be exploded, and verify that it is an array item array
    explode_col = get_overload_const_int(explode_col)
    if explode_col < 0 or explode_col >= len(in_arr_types):  # pragma: no cover
        raise_bodo_error(
            f"Invalid explode_col value for table with {len(in_arr_types)} columns: {explode_col}"
        )
    explode_col_arg = in_arr_types[explode_col]
    array_mode = False
    struct_mode = False
    map_mode = False
    if isinstance(explode_col_arg, ArrayItemArrayType):
        array_mode = True
    elif isinstance(explode_col_arg, MapArrayType):
        map_mode = True
    elif isinstance(explode_col_arg, StructArrayType):
        struct_mode = True

    if not (array_mode or map_mode or struct_mode):  # pragma: no cover
        raise_bodo_error(f"Invalid explode_col array type: {explode_col_arg}")

    # Create the tuple of columns that need to be kept when converting the input to a C++ table. This
    # includes all columns in keep_cols as well as the column that is to be exploded.
    keep_cols_tup = unwrap_typeref(keep_cols).meta
    in_col_inds = MetaType(
        (explode_col,)
        + tuple(i for i in range(len(in_arr_types)) if i in keep_cols_tup)
    )
    n_in_cols = len(in_col_inds.meta)

    outputs_tup = unwrap_typeref(outputs).meta
    if not isinstance(outputs_tup, tuple) or len(outputs_tup) != 6:  # pragma: no cover
        raise_bodo_error(f"lateral_flatten invalid outputs tuple: {outputs_tup}")
    (
        output_seq,
        output_key,
        output_path,
        output_index,
        output_val,
        output_this,
    ) = outputs_tup

    out_typs = tuple()

    output_seq_bool = get_overload_const_bool(output_seq)
    if output_seq_bool:  # pragma: no cover
        raise_bodo_error(
            f"lateral_flatten outputting value SEQ not currently supported"
        )

    output_key_bool = get_overload_const_bool(output_key)
    if output_key_bool:
        if struct_mode:
            out_typs += (bodo.dict_str_arr_type,)
        elif map_mode:
            out_typs += (explode_col_arg.key_arr_type,)
        else:  # pragma: no cover
            raise_bodo_error(
                f"lateral_flatten outputting value KEY not currently supported unless using"
            )

    output_path_bool = get_overload_const_bool(output_path)
    if output_path_bool:  # pragma: no cover
        raise_bodo_error(
            f"lateral_flatten outputting value PATH not currently supported"
        )

    # If the index column is included in the output, add an extra column to store it
    output_index_bool = get_overload_const_bool(output_index)
    if output_index_bool:
        out_typs += (bodo.IntegerArrayType(types.int64),)

    # If the value column is included in the output, add an extra column to store it
    output_val_bool = get_overload_const_bool(output_val)
    if output_val_bool:
        if struct_mode:
            common_type = get_combined_type(explode_col_arg.data, "flatten")
            out_typs += (common_type,)
        elif map_mode:
            out_typs += (explode_col_arg.value_arr_type,)
        else:
            out_typs += (explode_col_arg.dtype,)

    # If the 'this' column is included in the output, add an extra column to store it
    output_this_bool = get_overload_const_bool(output_this)
    if output_this_bool:  # pragma: no cover
        out_typs += (explode_col_arg,)

    # Add the arrays that are to be copied over during the explosion
    out_typs += tuple(
        in_arr_types[i] for i in range(len(in_arr_types)) if i in keep_cols_tup
    )

    out_col_inds = MetaType(tuple(list(range(len(out_typs)))))

    # Create the table type returned by the lateral operation
    out_types_0 = TableType(out_typs)
    out_types_1 = bodo.none
    n_out_table_cols = len(out_col_inds)

    json_mode = struct_mode or map_mode

    def impl(in_table, keep_cols, explode_col, outputs, outer):  # pragma: no cover
        # Create a single-element numpy array that C++ can use to store the number
        # of rows in the output table
        total_rows = np.array([0], dtype=np.int64)

        # Invoke the intrinsic to calculate the resulting table
        cpp_table = py_data_to_cpp_table(in_table, (), in_col_inds, n_in_cols)
        cpp_result = _lateral_flatten(
            cpp_table,
            total_rows.ctypes,
            output_seq_bool,
            output_key_bool,
            output_path_bool,
            output_index_bool,
            output_val_bool,
            output_this_bool,
            json_mode,
            outer,
        )
        bodo.utils.utils.check_and_propagate_cpp_exception()

        # Convert back to a Python table and cleanup any leftover tables
        py_result = cpp_table_to_py_data(
            cpp_result,
            out_col_inds,
            (out_types_0, out_types_1),
            total_rows[0],
            n_out_table_cols,
        )[0]
        delete_table(cpp_result)
        return py_result

    return impl
