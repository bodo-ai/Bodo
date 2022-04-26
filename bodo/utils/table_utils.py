# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""File containing utility functions for supporting DataFrame operations with Table Format."""

import numba
import numpy as np
from numba.core import types

import bodo
from bodo.hiframes.table import TableType
from bodo.utils.typing import (
    get_overload_const_str,
    is_overload_constant_str,
    is_overload_none,
    raise_bodo_error,
)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def generate_mappable_table_func(table, func_name, out_arr_typ, used_cols=None):
    """
    Function to help table implementations for mapping functions across
    column types and returning a Table of the same size.

    This function should be extended with more functionality as APIs are
    supported that require additional features (e.g. operations maintaining
    the same type per column, support for arguments, etc.).


    Keyword arguments:
    table -- Table upon which to map the function.
    func_name -- Name of the function being mapped to each column.
                 This must be a string literal.
                 Note: this can also be a function + ~.
    out_arr_typ -- Type of the output column, assumed to be a single
                   shared type for all columns
    used_cols -- Which columns are used/alive. If None presumed that every column
                 is alive. This is set in compiler optimizations and should not
                 be passed manually.
    """
    if not is_overload_constant_str(func_name):  # pragma: no cover
        raise_bodo_error(
            "generate_mappable_table_func(): func_name must be a constant string"
        )

    func_name = get_overload_const_str(func_name)
    out_typ = (
        out_arr_typ.instance_type
        if isinstance(out_arr_typ, types.TypeRef)
        else out_arr_typ
    )
    num_cols = len(table.arr_types)
    # Generate the typerefs for the output table
    col_typ_tuple = tuple([out_typ] * num_cols)
    table_typ = TableType(col_typ_tuple)
    # Set the globals
    glbls = {"bodo": bodo, "lst_dtype": out_typ, "table_typ": table_typ}

    func_text = "def impl(table, func_name, out_arr_typ, used_cols=None):\n"
    func_text += f"  out_list = bodo.hiframes.table.alloc_empty_list_type({num_cols}, lst_dtype)\n"
    # Convert used_cols to a set if it exists
    if not is_overload_none(used_cols):
        func_text += f"  used_cols_set = set(used_cols)\n"
    else:
        func_text += f"  used_cols_set = used_cols\n"

    # Select each block from the table
    func_text += f"  bodo.hiframes.table.ensure_table_unboxed(table, used_cols_set)\n"
    for blk in table.type_to_blk.values():
        func_text += f"  blk = bodo.hiframes.table.get_table_block(table, {blk})\n"
        # lower the original indices
        glbls[f"col_indices_{blk}"] = np.array(
            table.block_to_arr_ind[blk], dtype=np.int64
        )
        func_text += f"  for i in range(len(blk)):\n"
        # Since we have one list, store each element in its original column location
        func_text += f"    col_loc = col_indices_{blk}[i]\n"
        # Skip any dead columns
        if not is_overload_none(used_cols):
            func_text += f"    if col_loc not in used_cols_set:\n"
            func_text += f"        continue\n"
        # TODO: Support APIs that take additional arguments
        func_text += f"    out_list[col_loc] = {func_name}(blk[i])\n"
    func_text += (
        "  return bodo.hiframes.table.init_table_from_lists((out_list,), table_typ)"
    )

    local_vars = {}
    exec(func_text, glbls, local_vars)
    return local_vars["impl"]


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def generate_table_nbytes(table, out_arr, start_offset, parallel=False):
    """
    Function to compute nbytes on a table. Since nbytes requires a reduction
    across all columns and is most often used in initial testing, we assume
    there will never be any dead columns.

    Args:
    table: Input table. bytes will be computed across every array in this table.
    out_arr: Output array this is allocated in the calling function. This array
             will always be replicated and contains either n or n + 1 elements.
             The extra element will be if the index nbytes is also being computed,
             which is done outside of this function.
    start_offset: Either 0 or 1. If the array has n + 1 elements this will be 1 as
                  out_arr[0] should be the number of bytes used by the index
    parallel: Bodo internal argument for if the table is distributed. This should not
              be provided by a user and is set in distributed pass.

    Returns: None
    """
    # Set the globals
    glbls = {
        "bodo": bodo,
        "sum_op": np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value),
    }

    func_text = "def impl(table, out_arr, start_offset, parallel=False):\n"
    # Ensure the whole table is unboxed as we will use every column. Bodo loads
    # Tables/DataFrames from Python with "lazy" unboxing and some columns
    # may not be loaded yet.
    func_text += "  bodo.hiframes.table.ensure_table_unboxed(table, None)\n"
    # General code for each type stored inside the table.
    for blk in table.type_to_blk.values():
        func_text += f"  blk = bodo.hiframes.table.get_table_block(table, {blk})\n"
        # lower the original indices
        glbls[f"col_indices_{blk}"] = np.array(
            table.block_to_arr_ind[blk], dtype=np.int64
        )
        func_text += "  for i in range(len(blk)):\n"
        # Since we have one list, store each element in its original column location
        func_text += f"    col_loc = col_indices_{blk}[i]\n"
        func_text += "    out_arr[col_loc + start_offset] = blk[i].nbytes\n"
    # If we have parallel code do a reduction.
    func_text += "  if parallel:\n"
    func_text += "    for i in range(start_offset, len(out_arr)):\n"
    # TODO [BE-2614]: Do a reduction on the whole array at once
    func_text += (
        "      out_arr[i] = bodo.libs.distributed_api.dist_reduce(out_arr[i], sum_op)\n"
    )

    local_vars = {}
    exec(func_text, glbls, local_vars)
    return local_vars["impl"]
