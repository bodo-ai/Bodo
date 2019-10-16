# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Helper functions for transformations.
"""
import pandas as pd
import numpy as np
import math
import numba
from numba import ir, ir_utils, types
from numba.ir_utils import compile_to_numba_ir, replace_arg_nodes
import bodo
from bodo.utils.utils import is_assign, is_expr


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
    assert len(f_ir.blocks) == 1
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
