# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Helper information to keep table column deletion
pass organized. This contains information about all
table operations for optimizations.
"""
from typing import Dict, Tuple

from numba.core import ir, types

import bodo

# This must contain all table functions that can
# "use" a column. This is used by helper functions
# for pruning columns. Every table operation that
# either uses a column (e.g. get_table_data) or
# create a new table from the existing table
# (e.g. generate_table_nbytes) and can be exposed
# to the main IR must be included here. Operations
# that are aliases/reuse the same internal table
# (or its lists) without making copies should not
# be included here as they must be handled separately
# for correctness (e.g. set_table_data).
table_usecol_funcs = {
    ("get_table_data", "bodo.hiframes.table"),
    ("table_filter", "bodo.hiframes.table"),
    ("generate_mappable_table_func", "bodo.utils.table_utils"),
    ("table_astype", "bodo.utils.table_utils"),
    ("generate_table_nbytes", "bodo.utils.table_utils"),
    ("table_concat", "bodo.utils.table_utils"),
}


def is_table_use_column_ops(fdef: Tuple[str, str]):
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
    return fdef in table_usecol_funcs


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
        Optional[Sequence[int], None]: List of columns used by the operation
            or None if it uses every column.
    """
    if fdef == ("get_table_data", "bodo.hiframes.table"):
        col_num = typemap[call_expr.args[1].name].literal_value
        return [col_num]
    elif fdef in {
        ("table_filter", "bodo.hiframes.table"),
        ("table_astype", "bodo.utils.table_utils"),
        ("generate_mappable_table_func", "bodo.utils.table_utils"),
    }:
        kws = dict(call_expr.kws)
        if "used_cols" in kws:
            used_cols_var = kws["used_cols"]
            used_cols_typ = typemap[used_cols_var.name]
            used_cols_typ = used_cols_typ.instance_type
            if isinstance(used_cols_typ, bodo.utils.typing.MetaType):
                return used_cols_typ.meta
    elif fdef == ("table_concat", "bodo.utils.table_utils"):
        # Table concat passes the column numbers meta type
        # as argument 1.
        # TODO: Refactor to pass used_cols as a keyword
        # argument so this is consistent.
        used_cols_var = call_expr.args[1]
        used_cols_typ = typemap[used_cols_var.name]
        used_cols_typ = used_cols_typ.instance_type
        if isinstance(used_cols_typ, bodo.utils.typing.MetaType):
            return used_cols_typ.meta

    # If we don't have information about which columns this operation
    # kills, we return to None to indicate we must decref any remaining
    # columns that die in the current block. This is correct because we go
    # backwards through the IR.
    return None
