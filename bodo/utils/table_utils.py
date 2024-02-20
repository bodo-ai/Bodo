# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""File containing utility functions for supporting DataFrame operations with Table Format."""

from collections import defaultdict
from typing import Any, Dict, Set

import numba
import numpy as np
from numba.core import types
from numba.extending import overload
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.hiframes.table import TableType
from bodo.utils.typing import (
    MetaType,
    get_castable_arr_dtype,
    get_overload_const_bool,
    get_overload_const_str,
    is_overload_constant_bool,
    is_overload_constant_str,
    is_overload_false,
    is_overload_none,
    is_overload_true,
    raise_bodo_error,
    unwrap_typeref,
)
from bodo.utils.utils import set_wrapper


@numba.generated_jit(nopython=True, no_cpython_wrapper=True, no_unliteral=True)
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

    out_typ = unwrap_typeref(out_arr_typ)
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
    glbls = {
        "bodo": bodo,
        "lst_dtype": out_typ,
        "table_typ": table_typ,
        "set_wrapper": set_wrapper,
    }

    func_text = "def impl(table, func_name, out_arr_typ, is_method, used_cols=None):\n"
    if keep_input_typ:
        # We maintain the original types.
        func_text += f"  out_table = bodo.hiframes.table.init_table(table, False)\n"
        # XXX: Support changing length?
        func_text += f"  l = len(table)\n"
    else:
        # We are converting to a common type.
        func_text += f"  out_list = bodo.hiframes.table.alloc_empty_list_type({num_cols}, lst_dtype)\n"
    has_live_cols = True
    # Convert used_cols to a set if it exists
    if not is_overload_none(used_cols):
        # Get the actual MetaType from the TypeRef
        used_cols_type = used_cols.instance_type
        has_live_cols = len(used_cols_type) != 0
        used_cols_data = np.array(used_cols_type.meta, dtype=np.int64)
        glbls["used_cols_glbl"] = used_cols_data
        kept_blks = set([table.block_nums[i] for i in used_cols_data])
        func_text += f"  used_cols_set = set_wrapper(used_cols_glbl)\n"
    else:
        func_text += f"  used_cols_set = None\n"
        used_cols_data = None

    # Select each block from the table
    func_text += f"  bodo.hiframes.table.ensure_table_unboxed(table, used_cols_set)\n"
    for blk in table.type_to_blk.values():
        func_text += (
            f"  blk_{blk} = bodo.hiframes.table.get_table_block(table, {blk})\n"
        )
        if keep_input_typ:
            if has_live_cols:
                # If we are maintaining the same types, output each list.
                # Assign the name for the future loop as well.
                func_text += f"  out_list_{blk} = bodo.hiframes.table.alloc_list_like(blk_{blk}, len(blk_{blk}), False)\n"
                out_list_name = f"out_list_{blk}"
        else:
            out_list_name = "out_list"
        if used_cols_data is None or blk in kept_blks:
            # Only generate code is some column from the input blk is live
            func_text += f"  for i in range(len(blk_{blk})):\n"
            # lower the original indices
            glbls[f"col_indices_{blk}"] = np.array(
                table.block_to_arr_ind[blk], dtype=np.int64
            )
            # Since we have one list, store each element in its original column location
            func_text += f"    col_loc = col_indices_{blk}[i]\n"
            # Skip any dead columns
            if used_cols_data is not None:
                func_text += f"    if col_loc not in used_cols_set:\n"
                func_text += f"        continue\n"
            # TODO: Support APIs that take additional arguments
            if keep_input_typ:
                # If we are maintaining types reuse i
                out_idx_val = "i"
            else:
                out_idx_val = "col_loc"
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
        if keep_input_typ and has_live_cols:
            func_text += f"  out_table = bodo.hiframes.table.set_table_block(out_table, {out_list_name}, {blk})\n"

    if keep_input_typ:
        func_text += f"  out_table = bodo.hiframes.table.set_table_len(out_table, l)\n"
        func_text += "  return out_table\n"
    else:
        func_text += "  return bodo.hiframes.table.init_table_from_lists((out_list,), table_typ)\n"

    local_vars = {}
    exec(func_text, glbls, local_vars)
    return local_vars["impl"]


def generate_mappable_table_func_equiv(self, scope, equiv_set, loc, args, kws):
    """shape of generate_mappable_table_func is the same as the input"""
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_utils_table_utils_generate_mappable_table_func = (  # type: ignore
    generate_mappable_table_func_equiv
)


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
def table_concat(table, col_nums_meta, arr_type):
    """
    Concatenates the columns from table corresponding to col_nums, which will be a list of column numbers. Requires
    that all column data in col_nums columns are of arr_type, that all column data comes from the same block and
    that there are no duplicates in col_nums. Returns a data array of type arr_type.
    """
    arr_type = unwrap_typeref(arr_type)
    concat_blk = table.type_to_blk[arr_type]
    glbls: Dict[str, Any] = {"bodo": bodo}
    glbls["col_indices"] = np.array(table.block_to_arr_ind[concat_blk], dtype=np.int64)

    # Get the actual Metatype from the TypeRef
    col_nums = col_nums_meta.instance_type
    # Lower the col_nums as an array. We do this to keep the used columns
    # visible in the IR for table column deletion.
    glbls["col_nums"] = np.array(col_nums.meta, np.int64)
    func_text = "def impl(table, col_nums_meta, arr_type):\n"
    func_text += f"  blk = bodo.hiframes.table.get_table_block(table, {concat_blk})\n"
    func_text += (
        "  col_num_to_ind_in_blk = {c : i for i, c in enumerate(col_indices)}\n"
    )
    func_text += "  n = len(table)\n"
    is_string = arr_type == bodo.string_array_type
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


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def concat_tables(in_tables, used_cols=None):
    """Concatenate rows of all tables in a list of tables.

    Args:
        in_tables (types.List(TableType)): input list of tables
        used_cols (types.TypeRef(bodo.MetaType)
            | types.none): If not None, the columns that should be selected
            for concat. This is filled by the table column deletion steps.

    Returns:
        TableType: The concatenated table.
    """
    assert isinstance(in_tables, types.List) and isinstance(
        in_tables.dtype, TableType
    ), "concat_tables: list of TableType expected"

    table_type = in_tables.dtype

    glbls = {
        "bodo": bodo,
        "init_table": bodo.hiframes.table.init_table,
        "get_table_block": bodo.hiframes.table.get_table_block,
        "ensure_column_unboxed": bodo.hiframes.table.ensure_column_unboxed,
        "set_table_block": bodo.hiframes.table.set_table_block,
        "set_table_len": bodo.hiframes.table.set_table_len,
        "alloc_list_like": bodo.hiframes.table.alloc_list_like,
        "table_type": table_type,
        "set_wrapper": set_wrapper,
    }
    if not is_overload_none(used_cols):
        used_cols_type = used_cols.instance_type
        used_cols_data = np.array(used_cols_type.meta, dtype=np.int64)
        glbls["used_cols_vals"] = used_cols_data
        kept_blks = set([table_type.block_nums[i] for i in used_cols_data])
    else:
        used_cols_data = None

    func_text = "def table_concat_func(in_tables, used_cols=None):\n"
    func_text += f"  T2 = init_table(table_type, False)\n"
    func_text += f"  l = 0\n"

    # set table length using input lengths and return if no table column is used
    if used_cols_data is not None and len(used_cols_data) == 0:
        func_text += f"  for T in in_tables:\n"
        func_text += f"    l += len(T)\n"
        func_text += f"  T2 = set_table_len(T2, l)\n"
        func_text += f"  return T2\n"
        loc_vars = {}
        exec(func_text, glbls, loc_vars)
        return loc_vars["table_concat_func"]

    if used_cols_data is not None:
        func_text += f"  used_set = set_wrapper(used_cols_vals)\n"

    for blk in table_type.type_to_blk.values():
        # allocate output block array list
        func_text += f"  arr_list_{blk}_init = get_table_block(in_tables[0], {blk})\n"
        func_text += f"  out_arr_list_{blk} = alloc_list_like(arr_list_{blk}_init, len(arr_list_{blk}_init), False)\n"

        if used_cols_data is None or blk in kept_blks:
            # Check if anything from this type is live. If not skip generating
            # the loop code.
            glbls[f"arr_inds_{blk}"] = np.array(
                table_type.block_to_arr_ind[blk], dtype=np.int64
            )
            func_text += f"  for i in range(len(arr_inds_{blk})):\n"
            func_text += f"    arr_ind_{blk} = arr_inds_{blk}[i]\n"
            if used_cols_data is not None:
                func_text += f"    if arr_ind_{blk} not in used_set: continue\n"

            func_text += f"    in_arrs_{blk} = []\n"
            func_text += f"    for T_{blk} in in_tables:\n"
            func_text += f"      arr_list_{blk} = get_table_block(T_{blk}, {blk})\n"
            func_text += f"      ensure_column_unboxed(T_{blk}, arr_list_{blk}, i, arr_ind_{blk})\n"
            func_text += f"      in_arrs_{blk}.append(arr_list_{blk}[i])\n"

            func_text += (
                f"    out_arr_{blk} = bodo.libs.array_kernels.concat(in_arrs_{blk})\n"
            )
            func_text += f"    l = len(out_arr_{blk})\n"
            func_text += f"    out_arr_list_{blk}[i] = out_arr_{blk}\n"

        func_text += f"  T2 = set_table_block(T2, out_arr_list_{blk}, {blk})\n"
    func_text += f"  T2 = set_table_len(T2, l)\n"
    func_text += f"  return T2\n"

    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    return loc_vars["table_concat_func"]


@numba.generated_jit(nopython=True, no_cpython_wrapper=True, no_unliteral=True)
def table_astype(table, new_table_typ, copy, _bodo_nan_to_str, used_cols=None):
    """Table kernel to perform an astype operations, which
    converts (possibly) the input table to a new output table.
    This generates code per type pair (old_type, new_type) to
    try efficiently convert arrays. For more information on the
    overall design of this kernel, please refer to this
    confluence document:

    https://bodo.atlassian.net/wiki/spaces/B/pages/1069776999/Table+astype

    Args:
        table (TableType): Input table that needs to be converted
        new_table_typ (TypeRef(TableType)): Type that the output table
            should correspond to.
        copy (types.bool_ | types.BooleanLiteral): Should the arrays in
            the table be copied. Even arrays that do not have their type
            changed must obey this parameter.
            https://github.com/pandas-dev/pandas/blob/4bfe3d07b4858144c219b9346329027024102ab6/pandas/core/generic.py#L5896
        _bodo_nan_to_str (types.bool_ | types.BooleanLiteral): Should string NA
            values be a proper NA or the string NaN
        used_cols (types.Array(int64, 1, "C") | None): Array of columns that
            are actually used. If the kernel is inlined into the main IR, then this
            should not be passed directly. If used in the body of a custom IR node,
            then this can be passed directly. A default of None means all columns
            are used.

    Returns:
        TableType: The output table with types corresponding to new_table_typ.
    """

    # Get the underlying type for the typeref
    new_table_typ = unwrap_typeref(new_table_typ)

    # Optimized path for common case of no type change (used in streaming operators)
    if (
        table == new_table_typ
        and is_overload_false(copy)
        and is_overload_false(_bodo_nan_to_str)
        and is_overload_none(used_cols)
    ):
        return (
            lambda table, new_table_typ, copy, _bodo_nan_to_str, used_cols=None: table
        )  # pragma: no cover

    # Determine if we can avoid the copy
    may_copy = not is_overload_false(copy)
    must_copy = is_overload_true(copy)

    # Define the globals for this kernel. These
    # will be updated throughout codegen.
    glbls: Dict[str, Any] = {"bodo": bodo, "set_wrapper": set_wrapper}

    # Compute the types that must be copied. Every column
    # should be updated in the same location.
    old_arr_typs = table.arr_types
    new_arr_typs = new_table_typ.arr_types
    # Track the DF column numbers that changed
    changed_cols: Set[int] = set()
    # Track the type changes that occurs. We keep the output
    # types as the key to slightly simplify the code.
    changed_types: Dict[types.Type, Set[types.Type]] = defaultdict(set)
    # Which types contain columns that aren't converted
    kept_types: Set[types.Type] = set()
    for i, old_typ in enumerate(old_arr_typs):
        new_typ = new_arr_typs[i]
        if old_typ == new_typ:
            kept_types.add(old_typ)
        else:
            changed_cols.add(i)
            changed_types[new_typ].add(old_typ)

    # Generate the code
    func_text = "def impl_table_astype(table, new_table_typ, copy, _bodo_nan_to_str, used_cols=None):\n"

    # Allocate the new table and set its length
    func_text += f"  out_table = bodo.hiframes.table.init_table(new_table_typ, False)\n"
    func_text += (
        f"  out_table = bodo.hiframes.table.set_table_len(out_table, len(table))\n"
    )
    # Create the set of kept columns and changed cols.
    possible_cols = set(range(len(old_arr_typs)))
    copied_cols = possible_cols - changed_cols
    if not is_overload_none(used_cols):
        used_cols_type = used_cols.instance_type
        used_cols_set = set(used_cols_type.meta)
        # Only keep the columns that are selected via usecols
        changed_cols = changed_cols & used_cols_set
        copied_cols = copied_cols & used_cols_set
        # Note: the used columns are shared between input and output
        # because astype cannot reorder columns.
        kept_blks = set([table.block_nums[i] for i in used_cols_set])
    else:
        used_cols_set = None

    # create a list of columns to cast for each pair of input/output
    # "cast_cols_{old_block_num}_{new_block_num}" is the list of columns being cast from old_block_num
    # -> new_block_num
    changed_cols_dict = dict()
    for changed_col in changed_cols:
        old_block_num = table.block_nums[changed_col]
        new_block_num = new_table_typ.block_nums[changed_col]
        if f"cast_cols_{old_block_num}_{new_block_num}" in changed_cols_dict:
            changed_cols_dict[f"cast_cols_{old_block_num}_{new_block_num}"].append(
                changed_col
            )
        else:
            changed_cols_dict[f"cast_cols_{old_block_num}_{new_block_num}"] = [
                changed_col
            ]

    # Now use the generated lists, and create the sets
    # TODO: directly lower as sets when memory leak for lowering constant
    # sets is fixed.
    for orig_block_num in table.blk_to_type:
        for new_block_num in new_table_typ.blk_to_type:
            list_of_vals = changed_cols_dict.get(
                f"cast_cols_{orig_block_num}_{new_block_num}", []
            )
            glbls[f"cast_cols_{orig_block_num}_{new_block_num}"] = np.array(
                list(list_of_vals), dtype=np.int64
            )
            func_text += f"  cast_cols_{orig_block_num}_{new_block_num}_set = set_wrapper(cast_cols_{orig_block_num}_{new_block_num})\n"

    glbls["copied_cols"] = np.array(list(copied_cols), dtype=np.int64)
    func_text += f"  copied_cols_set = set_wrapper(copied_cols)\n"
    # Generate the initial blocks for the output table.
    for new_table_array_typ, new_table_blk in new_table_typ.type_to_blk.items():
        glbls[f"typ_list_{new_table_blk}"] = types.List(new_table_array_typ)
        func_text += f"  out_arr_list_{new_table_blk} = bodo.hiframes.table.alloc_list_like(typ_list_{new_table_blk}, {len(new_table_typ.block_to_arr_ind[new_table_blk])}, False)\n"
        if new_table_array_typ in kept_types:
            # Create the mapping from old table to new table
            orig_table_blk = table.type_to_blk[new_table_array_typ]
            if used_cols_set is None or orig_table_blk in kept_blks:
                # Skip any loops that have entirely dead blocks.
                old_idx = table.block_to_arr_ind[orig_table_blk]
                idxs = [new_table_typ.block_offsets[idx] for idx in old_idx]
                glbls[f"new_idx_{orig_table_blk}"] = np.array(idxs, np.int64)
                glbls[f"orig_arr_inds_{orig_table_blk}"] = np.array(old_idx, np.int64)
                # If some array may be/is copied, we need to iterate over the old block.
                func_text += f"  arr_list_{orig_table_blk} = bodo.hiframes.table.get_table_block(table, {orig_table_blk})\n"
                func_text += f"  for i in range(len(arr_list_{orig_table_blk})):\n"
                func_text += f"    arr_ind_{orig_table_blk} = orig_arr_inds_{orig_table_blk}[i]\n"
                func_text += (
                    f"    if arr_ind_{orig_table_blk} not in copied_cols_set:\n"
                )
                func_text += f"      continue\n"
                func_text += f"    bodo.hiframes.table.ensure_column_unboxed(table, arr_list_{orig_table_blk}, i, arr_ind_{orig_table_blk})\n"
                # Map to the new physical location.
                func_text += f"    out_idx_{new_table_blk}_{orig_table_blk} = new_idx_{orig_table_blk}[i]\n"
                func_text += (
                    f"    arr_val_{orig_table_blk} = arr_list_{orig_table_blk}[i]\n"
                )
                if must_copy:
                    func_text += f"    arr_val_{orig_table_blk} = arr_val_{orig_table_blk}.copy()\n"
                elif may_copy:
                    func_text += f"    arr_val_{orig_table_blk} = arr_val_{orig_table_blk}.copy() if copy else arr_val_{new_table_blk}\n"
                func_text += f"    out_arr_list_{new_table_blk}[out_idx_{new_table_blk}_{orig_table_blk}] = arr_val_{orig_table_blk}\n"

    # Generate the code for the types that must be converted.
    # Here we reuse the out_arr_list_{blk} from the previous
    # section and arr_list_{blk} if it was already loaded.
    # We add seen_types to track those input types that get
    # converted to another type.
    seen_types = set()
    for new_table_array_typ, new_table_blk in new_table_typ.type_to_blk.items():
        if new_table_array_typ in changed_types:
            glbls[f"typ_{new_table_blk}"] = get_castable_arr_dtype(new_table_array_typ)
            orig_types = changed_types[new_table_array_typ]
            for orig_typ in orig_types:
                orig_table_blk = table.type_to_blk[orig_typ]
                if used_cols_set is None or orig_table_blk in kept_blks:
                    # Skip any loops that have entirely dead blocks.
                    if orig_typ not in kept_types and orig_typ not in seen_types:
                        # Create the mapping from old table to new table if
                        # they don't already exist
                        old_idx = table.block_to_arr_ind[orig_table_blk]
                        idxs = [new_table_typ.block_offsets[idx] for idx in old_idx]
                        glbls[f"new_idx_{orig_table_blk}"] = np.array(idxs, np.int64)
                        glbls[f"orig_arr_inds_{orig_table_blk}"] = np.array(
                            old_idx, np.int64
                        )
                        func_text += f"  arr_list_{orig_table_blk} = bodo.hiframes.table.get_table_block(table, {orig_table_blk})\n"
                    seen_types.add(orig_typ)
                    func_text += f"  for i in range(len(arr_list_{orig_table_blk})):\n"
                    func_text += f"    arr_ind_{orig_table_blk} = orig_arr_inds_{orig_table_blk}[i]\n"
                    # For each column, in the orig_table_blk, check if it is being cast to this output type
                    func_text += f"    if arr_ind_{orig_table_blk} not in cast_cols_{orig_table_blk}_{new_table_blk}_set:\n"
                    func_text += f"      continue\n"
                    func_text += f"    bodo.hiframes.table.ensure_column_unboxed(table, arr_list_{orig_table_blk}, i, arr_ind_{orig_table_blk})\n"
                    # Map to the new physical location.
                    func_text += f"    out_idx_{new_table_blk}_{orig_table_blk} = new_idx_{orig_table_blk}[i]\n"
                    func_text += f"    arr_val_{new_table_blk} =  bodo.utils.conversion.fix_arr_dtype(arr_list_{orig_table_blk}[i], typ_{new_table_blk}, copy, nan_to_str=_bodo_nan_to_str, from_series=True)\n"
                    func_text += f"    out_arr_list_{new_table_blk}[out_idx_{new_table_blk}_{orig_table_blk}] = arr_val_{new_table_blk}\n"

        # Append the list to the output now that we are done with this output type.
        func_text += f"  out_table = bodo.hiframes.table.set_table_block(out_table, out_arr_list_{new_table_blk}, {new_table_blk})\n"
    func_text += "  return out_table\n"
    local_vars = {}
    exec(func_text, glbls, local_vars)
    return local_vars["impl_table_astype"]


def table_astype_equiv(self, scope, equiv_set, loc, args, kws):
    """shape of table_astype is the same as the input"""
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_utils_table_utils_table_astype = table_astype_equiv  # type: ignore


def drop_duplicates_table(
    data, ind_arr, ncols, keep_i, parallel=False
):  # pragma: no cover
    pass


@overload(drop_duplicates_table, no_unliteral=True)
def overload_drop_duplicates(in_table, ind_arr, ncols, keep_i, parallel=False):
    """
    Kernel implementation for drop_duplicates. ncols is the number of
    columns to check for possible duplicates, which are always at the front.
    keep_i is either 0, 1, or 2 corresponding to keep="first", "last", or False
    """

    # ncols is the number of columns that are checked for duplicates
    # ncols <= n_table_cols. The duplicate checked columns are always at the front.
    n_table_cols = len(in_table.arr_types)

    ignore_index = is_overload_none(ind_arr)

    extra_arrs = "()" if ignore_index else "(ind_arr,)"

    func_text = "def impl(in_table, ind_arr, ncols, keep_i, parallel=False):\n"
    func_text += f"  in_cpp_table = py_data_to_cpp_table(in_table, {extra_arrs}, in_col_inds, {n_table_cols})\n"
    # NOTE: C++ will delete table pointer
    func_text += "  out_cpp_table = drop_duplicates_cpp_table(in_cpp_table, parallel, ncols, keep_i, False, True)\n"
    func_text += "  out_table = cpp_table_to_py_table(out_cpp_table, table_idxs, py_table_type, 0)\n"

    if ignore_index:
        func_text += "  out_arr_index = None\n"
    else:
        func_text += f"  out_arr_index = array_from_cpp_table(out_cpp_table, {n_table_cols}, ind_arr)\n"
    func_text += "  delete_table(out_cpp_table)\n"
    func_text += "  return out_table, out_arr_index\n"
    loc_vars = {}
    exec(
        func_text,
        {
            "py_data_to_cpp_table": bodo.libs.array.py_data_to_cpp_table,
            "cpp_table_to_py_table": bodo.libs.array.cpp_table_to_py_table,
            "drop_duplicates_cpp_table": bodo.libs.array.drop_duplicates_cpp_table,
            "array_from_cpp_table": bodo.libs.array.array_from_cpp_table,
            "delete_table": bodo.libs.array.delete_table,
            "py_table_type": in_table,
            "in_col_inds": MetaType(
                tuple(range(n_table_cols + (0 if ignore_index else 1)))
            ),
            "table_idxs": np.arange(n_table_cols, dtype=np.int64),
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl
