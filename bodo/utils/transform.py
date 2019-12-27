# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Helper functions for transformations.
"""
import operator
import pandas as pd
import numpy as np
import math
import numba
from numba import ir, ir_utils, types
from numba.ir_utils import (
    compile_to_numba_ir,
    replace_arg_nodes,
    find_const,
    guard,
    GuardException,
    get_definition,
    require,
)
import bodo
from bodo.utils.utils import is_assign, is_expr
from bodo.libs.str_ext import string_type
from bodo.utils.typing import BodoError


def compile_func_single_block(func, args, ret_var, typing_info, extra_globals=None):
    """compiles functions that are just a single basic block.
    Does not handle defaults, freevars etc.
    typing_info is a structure that has typingctx, typemap, calltypes
    (could be the pass itself since not mutated).
    """
    # TODO: support recursive processing of compile function if necessary
    glbls = {"numba": numba, "np": np, "bodo": bodo, "pd": pd, "math": math}
    if extra_globals is not None:
        glbls.update(extra_globals)
    f_ir = compile_to_numba_ir(
        func,
        glbls,
        typing_info.typingctx,
        tuple(typing_info.typemap[arg.name] for arg in args),
        typing_info.typemap,
        typing_info.calltypes,
    )
    assert (
        len(f_ir.blocks) == 1
    ), "only single block functions supported in compile_func_single_block()"
    f_block = f_ir.blocks.popitem()[1]
    replace_arg_nodes(f_block, args)
    nodes = f_block.body[:-2]

    # update Loc objects, avoid changing input arg vars
    update_locs(nodes[len(args) :], typing_info.curr_loc)
    for stmt in nodes[: len(args)]:
        stmt.target.loc = typing_info.curr_loc

    if ret_var is not None:
        cast_assign = f_block.body[-2]
        assert is_assign(cast_assign) and is_expr(cast_assign.value, "cast")
        func_ret = cast_assign.value.value
        nodes.append(ir.Assign(func_ret, ret_var, typing_info.curr_loc))

    return nodes


def update_locs(node_list, loc):
    """Update Loc objects for list of generated statements
    """
    for stmt in node_list:
        stmt.loc = loc
        for v in stmt.list_vars():
            v.loc = loc
        if is_assign(stmt):
            stmt.value.loc = loc


def get_stmt_defs(stmt):
    if is_assign(stmt):
        return set([stmt.target.name])

    if type(stmt) in numba.analysis.ir_extension_usedefs:
        def_func = numba.analysis.ir_extension_usedefs[type(stmt)]
        _uses, defs = def_func(stmt)
        return defs

    return set()


def get_str_const_value(var, func_ir, err_msg, typemap=None, arg_types=None):
    """Get constant value of a variable if possible, otherwise raise error.
    If the variable is argument to the function, force recompilation with literal
    typing of the argument.
    """
    # literal type
    if typemap is not None:
        typ = typemap[var.name]
        if isinstance(typ, types.Literal):
            return typ.literal_value

    try:
        return find_const(func_ir, var)
    except GuardException:
        # if variable is argument, force literal
        var_def = guard(get_definition, func_ir, var)
        if isinstance(var_def, ir.Arg):
            # untyped passes can only pass arg_types, not typemap used above
            if arg_types is not None and isinstance(
                arg_types[var_def.index], types.Literal
            ):
                return arg_types[var_def.index].literal_value
            raise numba.errors.ForceLiteralArg({var_def.index}, loc=var.loc)

    raise BodoError(err_msg)


def find_str_const(func_ir, var, arg_types=None):
    """Check if a variable can be inferred as a string constant, and return
    the constant value, or raise GuardException otherwise.
    """
    # TODO: refactor this function to use get_str_const_value
    require(isinstance(var, ir.Var))
    var_def = get_definition(func_ir, var)
    if isinstance(var_def, (ir.Const, ir.Global, ir.FreeVar)):
        val = var_def.value
        require(isinstance(val, str))
        return val
    elif isinstance(var_def, ir.Arg) and arg_types is not None:
        arg_typ = arg_types[var_def.index]
        if isinstance(arg_typ, types.StringLiteral):
            val = arg_typ.literal_value
            return val
        # force literal only if argument is string
        if arg_typ == string_type:
            raise numba.errors.ForceLiteralArg({var_def.index}, loc=var.loc)

    # only add supported (s1+s2), TODO: extend to other expressions
    require(
        isinstance(var_def, ir.Expr)
        and var_def.op == "binop"
        and var_def.fn == operator.add
    )
    arg1 = find_str_const(func_ir, var_def.lhs)
    arg2 = find_str_const(func_ir, var_def.rhs)
    return arg1 + arg2
