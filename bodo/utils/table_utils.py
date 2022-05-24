# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""File containing utility functions for supporting DataFrame operations with Table Format."""

import numba
import numpy as np
from numba.core import types

import bodo
from bodo.hiframes.table import TableType
from bodo.utils.typing import (
    get_overload_const_bool,
    get_overload_const_str,
    is_overload_constant_bool,
    is_overload_constant_str,
    is_overload_none,
    raise_bodo_error,
)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def generate_mappable_table_func(
    table, func_name, out_arr_typ, is_method, used_cols=None
):
    """
    Function to help table implementations for mapping functions across
    column types and returning a Table of the same size.

    This function should be extended with more functionality as APIs are
    supported that require additional features (e.g. operations maintaining
    the same type per column, support for arguments, etc.).


    Keyword arguments:
    table -- Table upon which to map the function.
    func_name -- Name of the function being mapped to each column.
                 This must be a string literal or None (to make a table
                 copy).
                 Note: this can also be a function + ~.
    out_arr_typ -- Type of the output column, assumed to be a single
                   shared type for all columns. If `types.none` then we maintain
                   the existing type
    is_method -- Should the passed in name be called as a function or a method for each array.
    used_cols -- Which columns are used/alive. If None presumed that every column
                 is alive. This is set in compiler optimizations and should not
                 be passed manually.
    """
    if not is_overload_constant_str(func_name) and not is_overload_none(
        func_name
    ):  # pragma: no cover
        raise_bodo_error(
            "generate_mappable_table_func(): func_name must be a constant string"
        )
    if not is_overload_constant_bool(is_method):
        raise_bodo_error(
            "generate_mappable_table_func(): is_method must be a constant boolean"
        )

    # `func_name is None`` means we are just making a shallow
    # copy and not executing any function.
    has_func = not is_overload_none(func_name)
    if has_func:
        func_name = get_overload_const_str(func_name)
        is_method_const = get_overload_const_bool(is_method)
    out_typ = (
        out_arr_typ.instance_type
        if isinstance(out_arr_typ, types.TypeRef)
        else out_arr_typ
    )
    # Do we reuse the input data type
    keep_input_typ = out_typ == types.none
    num_cols = len(table.arr_types)
    # Generate the typerefs for the output table
    if keep_input_typ:
        table_typ = table
    else:
        col_typ_tuple = tuple([out_typ] * num_cols)
        table_typ = TableType(col_typ_tuple)
    # Set the globals
    glbls = {"bodo": bodo, "lst_dtype": out_typ, "table_typ": table_typ}

    func_text = "def impl(table, func_name, out_arr_typ, is_method, used_cols=None):\n"
    if keep_input_typ:
        # We maintain the original types.
        func_text += f"  out_table = bodo.hiframes.table.init_table(table, False)\n"
        # XXX: Support changing length?
        func_text += f"  l = len(table)\n"
    else:
        # We are converting to a common type.
        func_text += f"  out_list = bodo.hiframes.table.alloc_empty_list_type({num_cols}, lst_dtype)\n"
    # Convert used_cols to a set if it exists
    if not is_overload_none(used_cols):
        func_text += f"  used_cols_set = set(used_cols)\n"
    else:
        func_text += f"  used_cols_set = used_cols\n"

    # Select each block from the table
    func_text += f"  bodo.hiframes.table.ensure_table_unboxed(table, used_cols_set)\n"
    for blk in table.type_to_blk.values():
        func_text += (
            f"  blk_{blk} = bodo.hiframes.table.get_table_block(table, {blk})\n"
        )
        # lower the original indices
        glbls[f"col_indices_{blk}"] = np.array(
            table.block_to_arr_ind[blk], dtype=np.int64
        )
        if keep_input_typ:
            # If we are maintaining the same types, output each list.
            func_text += f"  out_list_{blk} = bodo.hiframes.table.alloc_list_like(blk_{blk}, False)\n"
        func_text += f"  for i in range(len(blk_{blk})):\n"
        # Since we have one list, store each element in its original column location
        func_text += f"    col_loc = col_indices_{blk}[i]\n"
        # Skip any dead columns
        if not is_overload_none(used_cols):
            func_text += f"    if col_loc not in used_cols_set:\n"
            func_text += f"        continue\n"
        # TODO: Support APIs that take additional arguments
        if keep_input_typ:
            # If we are maintaining types reuse i
            out_idx_val = "i"
            out_list_name = f"out_list_{blk}"
        else:
            out_idx_val = "col_loc"
            out_list_name = "out_list"
        if not has_func:
            func_text += f"    {out_list_name}[{out_idx_val}] = blk_{blk}[i]\n"
        elif is_method_const:
            func_text += (
                f"    {out_list_name}[{out_idx_val}] = blk_{blk}[i].{func_name}()\n"
            )
        else:
            func_text += (
                f"    {out_list_name}[{out_idx_val}] = {func_name}(blk_{blk}[i])\n"
            )
        if keep_input_typ:
            func_text += f"  out_table = bodo.hiframes.table.set_table_block(out_table, {out_list_name}, {blk})\n"

    if keep_input_typ:
        func_text += f"  out_table = bodo.hiframes.table.set_table_len(out_table, l)\n"
        func_text += "  return out_table"
    else:
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


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def table_concat(table, col_nums, arr_type):
    """
    Concatenates the columns from table corresponding to col_nums, which will be a list of column numbers. Requires
    that all column data in col_nums columns are of arr_type, that all column data comes from the same block and
    that there are no duplicates in col_nums. Returns a data array of type arr_type.
    """
    arr_type = (
        arr_type.instance_type if isinstance(arr_type, types.TypeRef) else arr_type
    )
    concat_blk = table.type_to_blk[arr_type]
    glbls = {"bodo": bodo}
    glbls["col_indices"] = np.array(table.block_to_arr_ind[concat_blk], dtype=np.int64)

    func_text = "def impl(table, col_nums, arr_type):\n"
    func_text += f"  blk = bodo.hiframes.table.get_table_block(table, {concat_blk})\n"
    func_text += (
        "  col_num_to_ind_in_blk = {c : i for i, c in enumerate(col_indices)}\n"
    )
    func_text += "  n = len(table)\n"
    is_string = bodo.utils.typing.is_str_arr_type(arr_type)
    if is_string:
        func_text += "  total_chars = 0\n"
        func_text += "  for c in col_nums:\n"
        func_text += "    bodo.hiframes.table.ensure_column_unboxed(table, blk, col_num_to_ind_in_blk[c], c)\n"
        func_text += "    arr = blk[col_num_to_ind_in_blk[c]]\n"
        func_text += "    total_chars += bodo.libs.str_arr_ext.num_total_chars(arr)\n"
        func_text += "  out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n * len(col_nums), total_chars)\n"
    else:
        func_text += "  out_arr = bodo.utils.utils.alloc_type(n * len(col_nums), arr_type, (-1,))\n"
    func_text += "  for i in range(len(col_nums)):\n"
    func_text += "    c = col_nums[i]\n"
    if not is_string:
        # unboxing must be done on first iteration over columns
        func_text += "    bodo.hiframes.table.ensure_column_unboxed(table, blk, col_num_to_ind_in_blk[c], c)\n"
    func_text += "    arr = blk[col_num_to_ind_in_blk[c]]\n"
    func_text += "    off = i * n\n"
    func_text += "    for j in range(len(arr)):\n"
    func_text += "      if bodo.libs.array_kernels.isna(arr, j):\n"
    func_text += "        bodo.libs.array_kernels.setna(out_arr, off+j)\n"
    func_text += "      else:\n"
    func_text += "        out_arr[off+j] = arr[j]\n"
    func_text += "  return out_arr\n"

    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    impl = loc_vars["impl"]

    return impl
