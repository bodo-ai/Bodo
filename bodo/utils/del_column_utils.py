# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Helper information to keep table column deletion
pass organized. This contains information about all
table operations for optimizations.
"""
from typing import Dict, Tuple

from numba.core import ir, types

from bodo.hiframes.table import TableType
from bodo.utils.typing import is_overload_none

# This must contain all table functions that can
# "use" a column. This is used by helper functions
# for pruning columns. Every table operation that
# either uses a column (e.g. get_table_data) or
# create a new table from the existing table
# (e.g. generate_table_nbytes) and can be exposed
# to the main IR must be included here.
table_usecol_funcs = {
    ("get_table_data", "bodo.hiframes.table"),
    ("table_filter", "bodo.hiframes.table"),
    ("table_local_filter", "bodo.hiframes.table"),
    ("table_subset", "bodo.hiframes.table"),
    ("set_table_data", "bodo.hiframes.table"),
    ("set_table_data_null", "bodo.hiframes.table"),
    ("generate_mappable_table_func", "bodo.utils.table_utils"),
    ("table_astype", "bodo.utils.table_utils"),
    ("generate_table_nbytes", "bodo.utils.table_utils"),
    ("table_concat", "bodo.utils.table_utils"),
    ("py_data_to_cpp_table", "bodo.libs.array"),
    ("logical_table_to_table", "bodo.hiframes.table"),
}


def is_table_use_column_ops(fdef: Tuple[str, str], args, typemap):
    """Is the given callname a table operation
    that uses columns. Note: This must include
    all valid table operations that do not result
    in `use_all` for an entire block.

    Args:
        fdef (Tuple[str, str]): Relevant call name

    Returns:
        Bool: Is the table a known operation that
            can produce a column deletion.
    """
    # Every function in table_usecol_funcs takes a table as
    # arg0.
    return (
        fdef in table_usecol_funcs
        and len(args) > 0
        and isinstance(typemap[args[0].name], TableType)
    )


def get_table_used_columns(
    fdef: Tuple[str, str], call_expr: ir.Expr, typemap: Dict[str, types.Type]
):
    """Get the columns used by a particular table operation

    Args:
        fdef (Tuple[str, str]): Relevant callname
        call_expr (ir.Expr): Call expresion
        typemap (Dict[str, types.Type]): Type map mapping variable names
            to types.

    Returns:
        Optional[Set[int], None]: Set of columns used by the operation
            or None if it uses every column.
    """
    if fdef == ("get_table_data", "bodo.hiframes.table"):
        col_num = typemap[call_expr.args[1].name].literal_value
        return {col_num}
    elif fdef in {
        ("table_filter", "bodo.hiframes.table"),
        ("table_local_filter", "bodo.hiframes.table"),
        ("table_astype", "bodo.utils.table_utils"),
        ("generate_mappable_table_func", "bodo.utils.table_utils"),
        ("set_table_data", "bodo.hiframes.table"),
        ("set_table_data_null", "bodo.hiframes.table"),
    }:
        kws = dict(call_expr.kws)
        if "used_cols" in kws:
            used_cols_var = kws["used_cols"]
            used_cols_typ = typemap[used_cols_var.name]
            # Double check that someone didn't manually specify
            # "none" for used_cols.
            if not is_overload_none(used_cols_typ):
                used_cols_typ = used_cols_typ.instance_type
                return set(used_cols_typ.meta)
    elif fdef == ("table_concat", "bodo.utils.table_utils"):
        # Table concat passes the column numbers meta type
        # as argument 1.
        # TODO: Refactor to pass used_cols as a keyword
        # argument so this is consistent.
        used_cols_var = call_expr.args[1]
        used_cols_typ = typemap[used_cols_var.name]
        used_cols_typ = used_cols_typ.instance_type
        return set(used_cols_typ.meta)
    elif fdef == ("table_subset", "bodo.hiframes.table"):
        # Table subset needs to map back to the original columns
        # via the idx subset in argument 1 and the "used_cols" kws.
        idx_var = call_expr.args[1]
        idx_typ = typemap[idx_var.name]
        idx_typ = idx_typ.instance_type
        idx_cols = idx_typ.meta
        # Determine if there are pruned columns,
        # which produces a different pass
        kws = dict(call_expr.kws)
        if "used_cols" in kws:
            # If there are used columns we need to remove
            # any idx values that are removed.
            used_cols_var = kws["used_cols"]
            used_cols_typ = typemap[used_cols_var.name]
            used_cols_typ = used_cols_typ.instance_type
            # These used cols are for the output table.
            # We need to return this to the input table.
            used_cols_set = set(used_cols_typ.meta)
            # Kept track of the orignal columns
            orig_used_cols = set()
            for i, orig_num in enumerate(idx_cols):
                if i in used_cols_set:
                    orig_used_cols.add(orig_num)
            return orig_used_cols
        else:
            # Remove any duplicate columns
            return set(idx_cols)

    elif fdef == ("py_data_to_cpp_table", "bodo.libs.array"):
        # py_data_to_cpp_table takes logical column indices of input table (arg 0) and
        # extra arrays (arg 1). Non-table indices need to be removed.
        used_cols = typemap[call_expr.args[2].name].instance_type.meta
        n_table_cols = len(typemap[call_expr.args[0].name].arr_types)
        return set(i for i in used_cols if i < n_table_cols)

    # NOTE: get_table_used_columns() is called only when first input of
    # logical_table_to_table() is a table
    elif fdef == ("logical_table_to_table", "bodo.hiframes.table"):
        in_col_inds = typemap[call_expr.args[2].name].instance_type.meta
        n_in_table_arrs = len(typemap[call_expr.args[0].name].arr_types)
        kws = dict(call_expr.kws)
        # remove columns that are dead in the output table
        if "used_cols" in kws:
            # map output column indices to input column indices
            used_cols_set = set(typemap[kws["used_cols"].name].instance_type.meta)
            in_used_cols = set()
            for out_ind, in_ind in enumerate(in_col_inds):
                if out_ind in used_cols_set and in_ind < n_in_table_arrs:
                    in_used_cols.add(in_ind)
            return in_used_cols
        else:
            return set(i for i in in_col_inds if i < n_in_table_arrs)

    # If we don't have information about which columns this operation
    # kills, we return to None to indicate we must decref any remaining
    # columns that die in the current block. This is correct because we go
    # backwards through the IR.
    return None
