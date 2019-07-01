"""
Helper functions for transformations.
"""
import pandas as pd
import numpy as np
import numba
from numba import ir, ir_utils, types
from numba.ir_utils import compile_to_numba_ir, replace_arg_nodes
import bodo
from bodo.utils.utils import is_assign, is_expr


def compile_func_single_block(func, args, ret_var, typing_info,
                                                           extra_globals=None):
    """compiles functions that are just a single basic block.
    Does not handle defaults, freevars etc.
    typing_info is a structure that has typingctx, typemap, calltypes
    (could be the pass itself since not mutated).
    """
    glbls = {'numba': numba, 'np': np, 'bodo': bodo, 'pd': pd}
    if extra_globals is not None:
        glbls.update(extra_globals)
    f_ir = compile_to_numba_ir(
        func, glbls, typing_info.typingctx,
        tuple(typing_info.typemap[arg.name] for arg in args),
        typing_info.typemap, typing_info.calltypes)
    assert len(f_ir.blocks) == 1
    f_block = f_ir.blocks.popitem()[1]
    replace_arg_nodes(f_block, args)
    nodes = f_block.body[:-2]
    if ret_var is not None:
        loc = ret_var.loc
        cast_assign = f_block.body[-2]
        assert is_assign(cast_assign) and is_expr(cast_assign.value, 'cast')
        func_ret = cast_assign.value.value
        nodes.append(ir.Assign(func_ret, ret_var, loc))
    return nodes
