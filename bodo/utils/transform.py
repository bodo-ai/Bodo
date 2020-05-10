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
import warnings

import numba
from numba.core import ir, ir_utils, types
from numba.core.ir_utils import (
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
from bodo.utils.typing import (
    BodoError,
    add_consts_to_registry,
)
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
    ("alloc_int_array", "int_arr_ext", "libs", bodo),
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
    ("integer_to_dt64", "pd_timestamp_ext", "hiframes", bodo),
    ("dt64_to_integer", "pd_timestamp_ext", "hiframes", bodo),
    ("timedelta64_to_integer", "pd_timestamp_ext", "hiframes", bodo),
    ("integer_to_timedelta64", "pd_timestamp_ext", "hiframes", bodo),
    ("npy_datetimestruct_to_datetime", "pd_timestamp_ext", "hiframes", bodo),
    # TODO: handle copy properly, copy of some types can have side effects?
    ("copy",),
    ("from_iterable_impl", "typing", "utils", bodo),
    ("chain", itertools),
    ("groupby",),
    ("rolling",),
    (pd.CategoricalDtype,),
    # Numpy
    ("int32", np),
    ("int64", np),
    ("float64", np),
    ("float32", np),
    ("bool_", np),
    ("full", np),
    ("round", np),
    ("isnan", np),
    ("isnat", np),
    # Numba
    ("internal_prange", "parfor", numba),
    ("internal_prange", "parfor", "parfors", numba),
    ("empty_inferred", "ndarray", "unsafe", numba),
    ("_slice_span", "unicode", numba),
    ("_normalize_slice", "unicode", numba),
    # hdf5
    ("h5size", "h5_api", "io", bodo),
    ("pre_alloc_list_string_array", "list_str_arr_ext", "libs", bodo),
    (bodo.libs.list_str_arr_ext.pre_alloc_list_string_array,),
    ("pre_alloc_list_item_array", "list_item_arr_ext", "libs", bodo),
    (bodo.libs.list_item_arr_ext.pre_alloc_list_item_array,),
    ("dist_reduce", "distributed_api", "libs", bodo),
    (bodo.libs.distributed_api.dist_reduce,),
    ("pre_alloc_string_array", "str_arr_ext", "libs", bodo),
    (bodo.libs.str_arr_ext.pre_alloc_string_array,),
    ("prange", bodo),
    (bodo.prange),
}


def remove_hiframes(rhs, lives, call_list):
    call_tuple = tuple(call_list)
    if call_tuple in no_side_effect_call_tuples:
        return True

    # TODO: probably not reachable here since always inlined?
    if len(call_list) == 4 and call_list[1:] == [
        "conversion",
        "utils",
        bodo,
    ]:  # pragma: no cover
        # all conversion functions are side effect-free
        return True

    # TODO: handle copy() of the relevant types properly
    if len(call_list) == 2 and call_list[0] == "copy":
        return True

    # TODO: probably not reachable here since only used in backend?
    if (
        call_list == [bodo.io.parquet_pio.read_parquet]
        and rhs.args[2].name not in lives
    ):  # pragma: no cover
        return True

    # can't add these to no_side_effect_call_tuples due to import issues, TODO: fix
    # TODO: probably not reachable here since only used in backend?
    if call_tuple in (
        (bodo.io.parquet_pio.get_column_size_parquet,),
        (bodo.io.parquet_pio.read_parquet_str,),
        (bodo.io.parquet_pio.read_parquet_list_str,),
    ):  # pragma: no cover
        return True

    # the call is dead if the read array is dead
    # TODO: return array from call to avoid using lives
    if call_list == ["h5read", "h5_api", "io", bodo] and rhs.args[5].name not in lives:
        return True

    if (
        call_list == ["move_str_arr_payload", "str_arr_ext", "libs", bodo]
        and rhs.args[0].name not in lives
    ):
        return True

    # TODO: needed?
    # if call_list == ['set_parent_dummy', 'pd_dataframe_ext', 'hiframes', bodo]:
    #     return True

    return False


numba.core.ir_utils.remove_call_handlers.append(remove_hiframes)


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

    if type(stmt) in numba.core.analysis.ir_extension_usedefs:
        def_func = numba.core.analysis.ir_extension_usedefs[type(stmt)]
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
        raise numba.core.errors.ForceLiteralArg({var_def.index}, loc=var.loc)

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

        f_ir = numba.core.ir_utils.get_ir_of_code(_globals, code)
    elif isinstance(func, bodo.utils.typing.FunctionLiteral):
        f_ir = numba.core.compiler.run_frontend(
            func.literal_value, inline_closures=True
        )
    else:
        assert isinstance(func, types.Dispatcher)
        py_func = func.dispatcher.py_func
        f_ir = numba.core.compiler.run_frontend(py_func, inline_closures=True)

    _, f_return_type, _ = numba.core.typed_passes.type_inference_stage(
        typing_context, f_ir, arg_types, None
    )
    return f_return_type


def update_node_list_definitions(node_list, func_ir):
    loc = ir.Loc("", 0)
    dumm_block = ir.Block(ir.Scope(None, loc), loc)
    dumm_block.body = node_list
    build_definitions({0: dumm_block}, func_ir._definitions)
    return


def gen_const_tup(vals):
    """generate a constant tuple value as text
    """
    return "({}{})".format(
        ", ".join("'{}'".format(c) if isinstance(c, str) else str(c) for c in vals),
        "," if len(vals) == 1 else ""
    )


def gen_add_consts_to_type_call(vals, var_name):
    """generate add_consts_to_type() call as text. Also returns the const object being
    preserved in the registry to enable the caller to keep a reference around.
    """
    const_obj, const_no = add_consts_to_registry(vals)
    func_call = "bodo.utils.typing.add_consts_to_type({}, {})".format(
        var_name, const_no
    )
    return const_obj, func_call


def gen_add_consts_to_type(vals, var, ret_var, typing_info=None):
    """generate add_consts_to_type() call that makes constant values of dict/list
    available during typing
    """

    const_obj, const_to_type_call = gen_add_consts_to_type_call(vals, "a")
    func_text = "def _build_f(a):\n"
    func_text += "  return {}\n".format(const_to_type_call)
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    _build_f = loc_vars["_build_f"]
    nodes = compile_func_single_block(_build_f, (var,), ret_var, typing_info)

    # HACK keep const values object around as long as the function is being compiled by
    # adding it as an attribute to some compilation object
    nodes[-1].const_obj = const_obj
    return nodes


def get_call_expr_arg(f_name, args, kws, arg_no, arg_name, default=None, err_msg=None):
    """get a specific argument from all argument variables of a call expr, which could
    be specified either as a positional argument or keyword argument.
    If argument is not specified, an error is raised unless if a default is specified.
    """
    arg = None
    if len(args) > arg_no:
        arg = args[arg_no]
    elif arg_name in kws:
        arg = kws[arg_name]

    if arg is None:
        if default is not None:
            return default
        if err_msg is None:
            err_msg = "{} requires '{}' argument".format(f_name, arg_name)
        raise BodoError(err_msg)
    return arg
