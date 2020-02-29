# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Helper functions for transformations.
"""
import operator
import math
import itertools
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
    find_callname,
    build_definitions,
)
import bodo
from bodo.utils.utils import is_assign, is_expr
from bodo.libs.str_ext import string_type
from bodo.utils.typing import BodoError
from bodo.utils.utils import is_call


no_side_effect_call_tuples = {
    # general python functions
    (int,),
    (list,),
    (set,),
    (dict,),
    (min,),
    (max,),
    (abs,),
    (len,),
    ("ceil", math),
    # Series
    ("init_series", "pd_series_ext", "hiframes", bodo),
    ("get_series_data", "pd_series_ext", "hiframes", bodo),
    ("get_series_index", "pd_series_ext", "hiframes", bodo),
    ("get_series_name", "pd_series_ext", "hiframes", bodo),
    ("convert_tup_to_rec", "typing", "utils", bodo),
    ("convert_rec_to_tup", "typing", "utils", bodo),
    ("internal_prange", "parfor", numba),
    # Index
    ("init_string_index", "pd_index_ext", "hiframes", bodo),
    ("init_numeric_index", "pd_index_ext", "hiframes", bodo),
    ("_dti_val_finalize", "pd_index_ext", "hiframes", bodo),
    ("init_datetime_index", "pd_index_ext", "hiframes", bodo),
    ("init_timedelta_index", "pd_index_ext", "hiframes", bodo),
    ("init_range_index", "pd_index_ext", "hiframes", bodo),
    # Int array
    ("get_int_arr_data", "int_arr_ext", "libs", bodo),
    ("get_int_arr_bitmap", "int_arr_ext", "libs", bodo),
    ("init_integer_array", "int_arr_ext", "libs", bodo),
    # bool array
    ("get_bool_arr_data", "bool_arr_ext", "libs", bodo),
    ("get_bool_arr_bitmap", "bool_arr_ext", "libs", bodo),
    ("init_bool_array", "bool_arr_ext", "libs", bodo),
    ("alloc_bool_array", "bool_arr_ext", "libs", bodo),
    ("alloc_datetime_date_array", "datetime_date_ext", "hiframes", bodo,),
    ("_sum_handle_nan", "series_kernels", "hiframes", bodo),
    ("_mean_handle_nan", "series_kernels", "hiframes", bodo),
    ("_var_handle_nan", "series_kernels", "hiframes", bodo),
    ("dist_return", "distributed_api", bodo),
    # dataframe
    ("init_dataframe", "pd_dataframe_ext", "hiframes", bodo),
    ("get_dataframe_data", "pd_dataframe_ext", "hiframes", bodo),
    ("get_dataframe_index", "pd_dataframe_ext", "hiframes", bodo),
    ("rolling_dummy", "pd_rolling_ext", "hiframes", bodo),
    # array kernels
    ("calc_nitems", "array_kernels", "libs", bodo),
    ("concat", "array_kernels", "libs", bodo),
    ("unique", "array_kernels", "libs", bodo),
    ("nunique", "array_kernels", "libs", bodo),
    ("quantile", "array_kernels", "libs", bodo),
    ("add_consts_to_type", "typing", "utils", bodo),
    ("str_arr_from_sequence", "str_arr_ext", "libs", bodo),
    ("parse_datetime_str", "pd_timestamp_ext", "hiframes", bodo),
    # TODO: handle copy properly, copy of some types can have side effects?
    ("copy",),
    ("from_iterable_impl", "typing", "utils", bodo),
    ("chain", itertools),
    ("empty_inferred", "ndarray", "unsafe", numba),
    ("groupby",),
    ("rolling",),
    (pd.CategoricalDtype,),
}


def remove_hiframes(rhs, lives, call_list):
    call_tuple = tuple(call_list)
    if call_tuple in no_side_effect_call_tuples:
        return True

    if len(call_list) == 4 and call_list[1:] == ["conversion", "utils", bodo]:
        # all conversion functions are side effect-free
        return True

    if len(call_list) == 2 and call_list[0] == "copy":
        return True

    # if call_list == ['set_parent_dummy', 'pd_dataframe_ext', 'hiframes', bodo]:
    #     return True

    return False


numba.ir_utils.remove_call_handlers.append(remove_hiframes)


def compile_func_single_block(
    func, args, ret_var, typing_info=None, extra_globals=None
):
    """compiles functions that are just a single basic block.
    Does not handle defaults, freevars etc.
    typing_info is a structure that has typingctx, typemap, calltypes
    (could be the pass itself since not mutated).
    """
    # TODO: support recursive processing of compile function if necessary
    glbls = {"numba": numba, "np": np, "bodo": bodo, "pd": pd, "math": math}
    if extra_globals is not None:
        glbls.update(extra_globals)
    loc = ir.Loc("", 0)
    if ret_var:
        loc = ret_var.loc
    if typing_info:
        loc = typing_info.curr_loc
        f_ir = compile_to_numba_ir(
            func,
            glbls,
            typing_info.typingctx,
            tuple(typing_info.typemap[arg.name] for arg in args),
            typing_info.typemap,
            typing_info.calltypes,
        )
    else:
        f_ir = compile_to_numba_ir(func, glbls)
    assert (
        len(f_ir.blocks) == 1
    ), "only single block functions supported in compile_func_single_block()"
    f_block = f_ir.blocks.popitem()[1]
    replace_arg_nodes(f_block, args)
    nodes = f_block.body[:-2]

    # update Loc objects, avoid changing input arg vars
    update_locs(nodes[len(args) :], loc)
    for stmt in nodes[: len(args)]:
        stmt.target.loc = loc

    if ret_var is not None:
        cast_assign = f_block.body[-2]
        assert is_assign(cast_assign) and is_expr(cast_assign.value, "cast")
        func_ret = cast_assign.value.value
        nodes.append(ir.Assign(func_ret, ret_var, loc))

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
    val = guard(find_str_const, func_ir, var, arg_types, typemap)
    if val is None:
        raise BodoError(err_msg)
    return val


def find_str_const(func_ir, var, arg_types=None, typemap=None):
    """Check if a variable can be inferred as a string constant, and return
    the constant value, or raise GuardException otherwise.
    """
    require(isinstance(var, ir.Var))
    var_def = get_definition(func_ir, var)

    # get type of variable if possible
    typ = None
    if typemap is not None:
        typ = typemap[var.name]
    if isinstance(var_def, ir.Arg) and arg_types is not None:
        typ = arg_types[var_def.index]

    # literal type
    if isinstance(typ, types.StringLiteral):
        return typ.literal_value

    # constant value
    if isinstance(var_def, (ir.Const, ir.Global, ir.FreeVar)):
        val = var_def.value
        require(isinstance(val, str))
        return val
    # argument dispatch, force literal only if argument is string
    elif isinstance(var_def, ir.Arg) and typ == string_type:
        raise numba.errors.ForceLiteralArg({var_def.index}, loc=var.loc)

    # only add supported (s1+s2), TODO: extend to other expressions
    require(
        isinstance(var_def, ir.Expr)
        and var_def.op == "binop"
        and var_def.fn == operator.add
    )
    arg1 = find_str_const(func_ir, var_def.lhs, arg_types, typemap)
    arg2 = find_str_const(func_ir, var_def.rhs, arg_types, typemap)
    return arg1 + arg2


def get_const_nested(func_ir, v):
    """get constant value for v, even if v is a constant list or set.
    Does not capture GuardException.
    """
    v_def = get_definition(func_ir, v)
    if is_call(v_def) and find_callname(func_ir, v_def) == (
        "add_consts_to_type",
        "bodo.utils.typing",
    ):
        v_def = get_definition(func_ir, v_def.args[0])
    if isinstance(v_def, ir.Expr) and v_def.op in (
        "build_list",
        "build_set",
        "build_tuple",
    ):
        return tuple(get_const_nested(func_ir, a) for a in v_def.items)
    # treat make_function exprs as constant
    if is_expr(v_def, "make_function"):  # pragma: no cover
        return v_def
    return find_const(func_ir, v)


def get_const_func_output_type(func, arg_types, typing_context):
    """Get output type of constant function 'func' when compiled with 'arg_types' as
    argument types.
    'func' can be a MakeFunctionLiteral (inline lambda) or FunctionLiteral (function)
    """

    # MakeFunctionLiteral is not possible currently due to Numba's
    # MakeFunctionToJitFunction pass but may be possible later
    if isinstance(func, types.MakeFunctionLiteral):  # pragma: no cover
        code = func.literal_value.code
        _globals = {"np": np, "pd": pd, "numba": numba, "bodo": bodo}
        # XXX hack in untyped_pass to make globals available
        if hasattr(func.literal_value, "globals"):
            # TODO: use code.co_names to find globals actually used?
            _globals = func.literal_value.globals

        f_ir = numba.ir_utils.get_ir_of_code(_globals, code)
    elif isinstance(func, bodo.utils.typing.FunctionLiteral):
        f_ir = numba.compiler.run_frontend(func.literal_value, inline_closures=True)
    else:
        assert isinstance(func, types.Dispatcher)
        py_func = func.dispatcher.py_func
        f_ir = numba.compiler.run_frontend(py_func, inline_closures=True)

    _, f_return_type, _ = numba.typed_passes.type_inference_stage(
        typing_context, f_ir, arg_types, None
    )
    return f_return_type


def update_node_list_definitions(node_list, func_ir):
    loc = ir.Loc("", 0)
    dumm_block = ir.Block(ir.Scope(None, loc), loc)
    dumm_block.body = node_list
    build_definitions({0: dumm_block}, func_ir._definitions)
    return


def gen_add_consts_to_type(vals, var, ret_var, typing_info=None):
    """generate add_consts_to_type() call that makes constant values of dict/list
    available during typing
    """
    # convert constants to string representation
    const_funcs = {}
    val_reps = []
    for c in vals:
        v_rep = "{}".format(c)
        if isinstance(c, str):
            v_rep = "'{}'".format(c)
        # store a name for make_function exprs to replace later
        elif is_expr(c, "make_function") or isinstance(
            c, numba.targets.registry.CPUDispatcher
        ):
            v_rep = "func{}".format(ir_utils.next_label())
            const_funcs[v_rep] = c
        val_reps.append(v_rep)

    vals_expr = ", ".join(val_reps)
    func_text = "def _build_f(a):\n"
    func_text += "  return bodo.utils.typing.add_consts_to_type(a, {})\n".format(
        vals_expr
    )
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    _build_f = loc_vars["_build_f"]
    nodes = compile_func_single_block(_build_f, (var,), ret_var, typing_info)
    # replace make_function exprs with actual node
    for stmt in nodes:
        if (
            is_assign(stmt)
            and isinstance(stmt.value, ir.Global)
            and stmt.value.name in const_funcs
        ):
            v = const_funcs[stmt.value.name]
            if is_expr(v, "make_function"):
                stmt.value = const_funcs[stmt.value.name]
            # CPUDispatcher case
            else:
                stmt.value.value = const_funcs[stmt.value.name]
    return nodes
