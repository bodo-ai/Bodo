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
import numba.targets.linalg
from numba.targets.imputils import impl_ret_new_ref

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
    # Numba
    ("internal_prange", "parfor", numba),
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

    # TODO: needed?
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
        # if c is a python class (like str, float, int, list),
        # v_rep should be the class name
        if isinstance(c, type):
            v_rep = c.__name__
        elif isinstance(c, str):
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


# `run_frontend` function of Numba is used in inline_closure_call to get the IR of the
# function to be inlined.
# The code below is copied from Numba and modified to handle 'raise' nodes by running
# rewrite passes before inlining (feature copied from numba.ir_utils.get_ir_of_code).
# usecase example: bodo/tests/test_series.py::test_series_combine"[S13-S23-None-True]"
# https://github.com/numba/numba/blob/cc7e7c7cfa6389b54d3b5c2c95751c97eb531a96/numba/compiler.py#L186
def run_frontend(func, inline_closures=False):
    """
    Run the compiler frontend over the given Python function, and return
    the function's canonical Numba IR.

    If inline_closures is Truthy then closure inlining will be run
    """
    # XXX make this a dedicated Pipeline?
    func_id = numba.bytecode.FunctionIdentity.from_function(func)
    interp = numba.interpreter.Interpreter(func_id)
    bc = numba.bytecode.ByteCode(func_id=func_id)
    func_ir = interp.interpret(bc)
    if inline_closures:
        # code added to original 'run_frontend' to add rewrite passes
        # we need to run the before inference rewrite pass to normalize the IR
        # XXX: check rewrite pass flag?
        # for example, Raise nodes need to become StaticRaise before type inference
        class DummyPipeline:
            def __init__(self, f_ir):
                self.state = numba.compiler.StateDict()
                self.state.typingctx = None
                self.state.targetctx = None
                self.state.args = None
                self.state.func_ir = f_ir
                self.state.typemap = None
                self.state.return_type = None
                self.state.calltypes = None

        numba.rewrites.rewrite_registry.apply(
            "before-inference", DummyPipeline(func_ir).state
        )
        inline_pass = numba.inline_closurecall.InlineClosureCallPass(
            func_ir, numba.targets.cpu.ParallelOptions(False), {}, False
        )
        inline_pass.run()
    post_proc = numba.postproc.PostProcessor(func_ir)
    post_proc.run()
    return func_ir


numba.compiler.run_frontend = run_frontend


# The code below is copied from Numba and modified to handle aliases with tuple values.
# https://github.com/numba/numba/blob/cc7e7c7cfa6389b54d3b5c2c95751c97eb531a96/numba/ir_utils.py#L725
# This case happens for Bodo dataframes since init_dataframe takes a tuple of arrays as
# input, and output dataframe is aliased with all of these arrays. see test_df_alias.
from numba.ir_utils import _add_alias, alias_analysis_extensions, alias_func_extensions
import copy


# immutable scalar types, no aliasing possible
_immutable_type_class = (
    types.Number,
    types.scalars._NPDatetimeBase,
    types.iterators.RangeType,
    types.UnicodeType,
)


def is_immutable_type(var, typemap):
    # Conservatively, assume mutable if type not available
    if typemap is None or var not in typemap:
        return False
    typ = typemap[var]

    # TODO: add more immutable types
    if isinstance(typ, _immutable_type_class):
        return True

    if isinstance(typ, types.BaseTuple) and all(
        isinstance(t, _immutable_type_class) for t in typ.types
    ):
        return True
    # consevatively, assume mutable
    return False


def find_potential_aliases(
    blocks, args, typemap, func_ir, alias_map=None, arg_aliases=None
):
    "find all array aliases and argument aliases to avoid remove as dead"
    if alias_map is None:
        alias_map = {}
    if arg_aliases is None:
        arg_aliases = set(a for a in args if not is_immutable_type(a, typemap))

    # update definitions since they are not guaranteed to be up-to-date
    # FIXME keep definitions up-to-date to avoid the need for rebuilding
    func_ir._definitions = build_definitions(func_ir.blocks)
    np_alias_funcs = ["ravel", "transpose", "reshape"]

    for bl in blocks.values():
        for instr in bl.body:
            if type(instr) in alias_analysis_extensions:
                f = alias_analysis_extensions[type(instr)]
                f(instr, args, typemap, func_ir, alias_map, arg_aliases)
            if isinstance(instr, ir.Assign):
                expr = instr.value
                lhs = instr.target.name
                # only mutable types can alias
                if is_immutable_type(lhs, typemap):
                    continue
                if isinstance(expr, ir.Var) and lhs != expr.name:
                    _add_alias(lhs, expr.name, alias_map, arg_aliases)
                # subarrays like A = B[0] for 2D B
                if isinstance(expr, ir.Expr) and (
                    expr.op == "cast" or expr.op in ["getitem", "static_getitem"]
                ):
                    _add_alias(lhs, expr.value.name, alias_map, arg_aliases)
                # array attributes like A.T
                if (
                    isinstance(expr, ir.Expr)
                    and expr.op == "getattr"
                    and expr.attr in ["T", "ctypes", "flat"]
                ):
                    _add_alias(lhs, expr.value.name, alias_map, arg_aliases)
                # a = b.c.  a should alias b
                if (
                    isinstance(expr, ir.Expr)
                    and expr.op == "getattr"
                    and expr.value.name in arg_aliases
                ):
                    _add_alias(lhs, expr.value.name, alias_map, arg_aliases)
                # new code added to handle tuple/list/set of mutable data
                if (
                    isinstance(expr, ir.Expr)
                    and expr.op in ("build_tuple", "build_list", "build_set")
                    and not is_immutable_type(lhs, typemap)
                ):
                    for v in expr.items:
                        _add_alias(lhs, v.name, alias_map, arg_aliases)
                # calls that can create aliases such as B = A.ravel()
                if isinstance(expr, ir.Expr) and expr.op == "call":
                    fdef = guard(find_callname, func_ir, expr, typemap)
                    # TODO: sometimes gufunc backend creates duplicate code
                    # causing find_callname to fail. Example: test_argmax
                    # ignored here since those cases don't create aliases
                    # but should be fixed in general
                    if fdef is None:
                        continue
                    fname, fmod = fdef
                    if fdef in alias_func_extensions:
                        alias_func = alias_func_extensions[fdef]
                        alias_func(lhs, expr.args, alias_map, arg_aliases)
                    if fmod == "numpy" and fname in np_alias_funcs:
                        _add_alias(lhs, expr.args[0].name, alias_map, arg_aliases)
                    if isinstance(fmod, ir.Var) and fname in np_alias_funcs:
                        _add_alias(lhs, fmod.name, alias_map, arg_aliases)

    # copy to avoid changing size during iteration
    old_alias_map = copy.deepcopy(alias_map)
    # combine all aliases transitively
    for v in old_alias_map:
        for w in old_alias_map[v]:
            alias_map[v] |= alias_map[w]
        for w in old_alias_map[v]:
            alias_map[w] = alias_map[v]

    return alias_map, arg_aliases


ir_utils.find_potential_aliases = find_potential_aliases


# The code below is copied from Numba and modified to fix Numba #5539.
# TODO: remove when the issue is fixed
# https://github.com/numba/numba/blob/afd5c67b1ed6f51c040d1845a014abea8b87846a/numba/np/linalg.py#L462
def dot_2_vm(context, builder, sig, args):
    """
    np.dot(vector, matrix)
    """
    def dot_impl(a, b):  # pragma: no cover
        m, = a.shape
        _m, n = b.shape
        # changed code: initialize with zeros if inputs are empty
        if m == 0:
            out = np.zeros((n, ), a.dtype)
        else:
            out = np.empty((n, ), a.dtype)
        return np.dot(a, b, out)

    res = context.compile_internal(builder, dot_impl, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)


numba.targets.linalg.dot_2_vm = dot_2_vm
