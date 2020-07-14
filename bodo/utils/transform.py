# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Helper functions for transformations.
"""
import operator
from collections import namedtuple
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
    mk_unique_var,
    compute_cfg_from_blocks,
)
from numba.core.registry import CPUDispatcher

import bodo
from bodo.utils.utils import is_assign, is_expr
from bodo.libs.str_ext import string_type
from bodo.utils.typing import (
    BodoError,
    BodoConstUpdatedError,
    can_literalize_type,
    is_literal_type,
    get_literal_value,
    get_overload_const_list,
)
from bodo.utils.utils import is_call, is_array_typ
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.struct_arr_ext import StructArrayType, StructType
from bodo.libs.array_item_arr_ext import ArrayItemArrayType


ReplaceFunc = namedtuple(
    "ReplaceFunc", ["func", "arg_types", "args", "glbls", "pre_nodes"]
)


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
    ("_var_handle_mincount", "series_kernels", "hiframes", bodo),
    ("_handle_nan_count", "series_kernels", "hiframes", bodo),
    ("_handle_nan_count_ddof", "series_kernels", "hiframes", bodo),
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
    ("pre_alloc_struct_array", "struct_arr_ext", "libs", bodo),
    (bodo.libs.struct_arr_ext.pre_alloc_struct_array,),
    ("pre_alloc_array_item_array", "array_item_arr_ext", "libs", bodo),
    (bodo.libs.array_item_arr_ext.pre_alloc_array_item_array,),
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
        (bodo.io.parquet_pio.read_parquet_array_item,),
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


def get_const_value(var, func_ir, err_msg, typemap=None, arg_types=None):
    """Get constant value of a variable if possible, otherwise raise error.
    If the variable is argument to the function, force recompilation with literal
    typing of the argument.
    """
    try:
        val = get_const_value_inner(func_ir, var, arg_types, typemap)
    except GuardException:
        raise BodoError(err_msg)
    return val


def get_const_value_inner(
    func_ir, var, arg_types=None, typemap=None, updated_containers=None
):
    """Check if a variable can be inferred as a constant and return the constant value.
    Otherwise, raise GuardException.
    """
    require(isinstance(var, ir.Var))
    var_def = get_definition(func_ir, var)

    # get type of variable if possible
    typ = None
    if typemap is not None:
        typ = typemap.get(var.name, None)
    if isinstance(var_def, ir.Arg) and arg_types is not None:
        typ = arg_types[var_def.index]

    # literal type case
    if is_literal_type(typ):
        return get_literal_value(typ)

    # constant value
    if isinstance(var_def, (ir.Const, ir.Global, ir.FreeVar)):
        val = var_def.value
        return val

    # argument dispatch, force literal only if argument can be literal
    if isinstance(var_def, ir.Arg) and can_literalize_type(typ):
        raise numba.core.errors.ForceLiteralArg({var_def.index}, loc=var.loc)

    # binary op (s1 op s2)
    if is_expr(var_def, "binop"):
        arg1 = get_const_value_inner(func_ir, var_def.lhs, arg_types, typemap)
        arg2 = get_const_value_inner(func_ir, var_def.rhs, arg_types, typemap)
        return var_def.fn(arg1, arg2)

    # df.columns case
    if is_expr(var_def, "getattr") and typemap:
        obj_typ = typemap.get(var_def.value.name, None)
        if isinstance(obj_typ, bodo.hiframes.pd_dataframe_ext.DataFrameType):
            return obj_typ.columns

    # list/set/dict cases

    # try dict.keys()
    call_name = guard(find_callname, func_ir, var_def)
    if (
        call_name is not None
        and len(call_name) == 2
        and call_name[0] == "keys"
        and isinstance(call_name[1], ir.Var)
    ):
        call_func = var_def.func
        var_def = get_definition(func_ir, call_name[1])
        dict_varname = call_name[1].name
        if updated_containers and dict_varname in updated_containers:
            raise BodoConstUpdatedError(
                "variable '{}' is updated inplace using '{}'".format(
                    dict_varname, updated_containers[dict_varname]
                )
            )

        require(is_expr(var_def, "build_map"))
        vals = [v[0] for v in var_def.items]
        # HACK replace dict.keys getattr to avoid typing errors
        keys_getattr = guard(get_definition, func_ir, call_func)
        assert isinstance(keys_getattr, ir.Expr) and keys_getattr.attr == "keys"
        keys_getattr.attr = "copy"
        return [get_const_value_inner(func_ir, v, arg_types, typemap) for v in vals]

    if updated_containers and var.name in updated_containers:
        raise BodoConstUpdatedError(
            "variable '{}' is updated inplace using '{}'".format(
                var.name, updated_containers[var.name]
            )
        )

    # dict case
    if is_expr(var_def, "build_map"):
        return {
            get_const_value_inner(
                func_ir, v[0], arg_types, typemap
            ): get_const_value_inner(func_ir, v[1], arg_types, typemap)
            for v in var_def.items
        }

    # tuple case
    if is_expr(var_def, "build_tuple"):
        return tuple(
            get_const_value_inner(func_ir, v, arg_types, typemap) for v in var_def.items
        )

    # list
    if is_expr(var_def, "build_list"):
        return [
            get_const_value_inner(func_ir, v, arg_types, typemap) for v in var_def.items
        ]

    # list() call
    if call_name == ("list", "builtins"):
        values = get_const_value_inner(func_ir, var_def.args[0], arg_types, typemap)
        # sort set values when converting to list to have consistent order across
        # processors (e.g. important for join keys, see test_merge_multi_int_key)
        if isinstance(values, set):
            values = sorted(values)
        return list(values)

    # set() call
    if call_name == ("set", "builtins"):
        return set(get_const_value_inner(func_ir, var_def.args[0], arg_types, typemap))

    raise GuardException("Constant value not found")


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
    elif isinstance(func, CPUDispatcher):
        py_func = func.py_func
        f_ir = numba.core.compiler.run_frontend(py_func, inline_closures=True)
    else:
        assert isinstance(func, types.Dispatcher)
        py_func = func.dispatcher.py_func
        f_ir = numba.core.compiler.run_frontend(py_func, inline_closures=True)

    struct_key_names = fix_struct_return(f_ir)
    _, f_return_type, _ = numba.core.typed_passes.type_inference_stage(
        typing_context, f_ir, arg_types, None
    )
    # replace returned dictionary with a StructType to enabling typing for
    # StructArrayType later
    if isinstance(f_return_type, types.DictType) and struct_key_names:
        f_return_type = StructType(
            (f_return_type.value_type,) * len(struct_key_names), struct_key_names
        )
    return f_return_type


def _replace_const_map_return(f_ir, block, label):
    """replaces constant dictionary return value with a struct if values are not
    homogeneous, e.g. {"A": 1, "B": 2.3} -> struct((1, 2.3), ("A", "B"))
    """
    # get const map in return
    require(isinstance(block.body[-1], ir.Return))
    return_val = block.body[-1].value
    cast_def = guard(get_definition, f_ir, return_val)
    require(is_expr(cast_def, "cast"))
    ret_def = guard(get_definition, f_ir, cast_def.value)
    require(is_expr(ret_def, "build_map"))
    require(len(ret_def.items) > 0)
    keys = []
    key_strs = []
    values = []
    for (k, v) in ret_def.items:
        k_str = find_const(f_ir, k)
        require(isinstance(k_str, str))
        key_strs.append(k_str)
        keys.append(k)
        values.append(v)

    # {"A": v1, "B": v2} -> struct_if_heter_dict((v1, v2), ("A", "B"))
    loc = block.loc
    scope = block.scope
    # val_tup = (v1, v2)
    val_tup = ir.Var(scope, mk_unique_var("val_tup"), loc)
    val_tup_assign = ir.Assign(ir.Expr.build_tuple(values, loc), val_tup, loc)
    f_ir._definitions[val_tup.name] = [val_tup_assign.value]
    # key_tup = ("A", "B")
    key_tup = ir.Var(scope, mk_unique_var("key_tup"), loc)
    key_tup_assign = ir.Assign(ir.Expr.build_tuple(keys, loc), key_tup, loc)
    f_ir._definitions[key_tup.name] = [key_tup_assign.value]
    # new_Var = struct_if_heter_dict(val_tup, key_tup)
    call_var = ir.Var(scope, mk_unique_var("conv_call"), loc)
    call_global = ir.Assign(
        ir.Global(
            "struct_if_heter_dict", bodo.utils.conversion.struct_if_heter_dict, loc
        ),
        call_var,
        loc,
    )
    f_ir._definitions[call_var.name] = [call_global.value]
    new_var = ir.Var(scope, mk_unique_var("struct_val"), loc)
    new_assign = ir.Assign(
        ir.Expr.call(call_var, [val_tup, key_tup], {}, loc), new_var, loc
    )
    f_ir._definitions[new_var.name] = [new_assign.value]
    cast_def.value = new_var
    # {"A": v1, "B": v2} -> {"A": "A", "B": "B"} to avoid typing errors
    ret_def.items = [(k, k) for (k, _) in ret_def.items]
    block.body = (
        block.body[:-2]
        + [val_tup_assign, key_tup_assign, call_global, new_assign]
        + block.body[-2:]
    )
    return tuple(key_strs)


def fix_struct_return(f_ir):
    """replaces constant dictionary return value with a struct for all return blocks
    in 'f_ir'. Returns the key names if output is a struct.
    """
    key_names = None
    cfg = compute_cfg_from_blocks(f_ir.blocks)
    for exit_label in cfg.exit_points():
        key_names = guard(
            _replace_const_map_return, f_ir, f_ir.blocks[exit_label], exit_label
        )
    return key_names


def update_node_list_definitions(node_list, func_ir):
    loc = ir.Loc("", 0)
    dumm_block = ir.Block(ir.Scope(None, loc), loc)
    dumm_block.body = node_list
    build_definitions({0: dumm_block}, func_ir._definitions)
    return


# sentinel for nested const tuple gen used below
NESTED_TUP_SENTINEL = "$BODO_NESTED_TUP"


def gen_const_val_str(c):
    """convert value 'c' to string constant
    """
    # const nested constant tuples are not supported in Numba yet, need special handling
    # HACK: flatten tuple values but add a sentinel value that specifies how many
    # elements are from the nested tuple. Supports only one level nesting
    # TODO: fix nested const tuple handling in Numba
    if isinstance(c, tuple):
        return "'{}{}', ".format(NESTED_TUP_SENTINEL, len(c)) + ", ".join(
            gen_const_val_str(v) for v in c
        )
    return "'{}'".format(c) if isinstance(c, str) else str(c)


def gen_const_tup(vals):
    """generate a constant tuple value as text
    """
    val_seq = ", ".join(gen_const_val_str(c) for c in vals)
    return "({}{})".format(val_seq, "," if len(vals) == 1 else "",)


def get_const_tup_vals(c_typ):
    """get constant values from a tuple type generated using 'gen_const_tup'
    reverses the hack in 'gen_const_val_str'
    """
    vals = get_overload_const_list(c_typ)
    out = []
    i = 0
    while i < len(vals):
        v = vals[i]
        # reverse nested tuple flattening in gen_const_val_str
        if isinstance(v, str) and v.startswith(NESTED_TUP_SENTINEL):
            n_elem = int(v[len(NESTED_TUP_SENTINEL) :])
            out.append(tuple(vals[i + 1 : i + n_elem + 1]))
            i += n_elem + 1
        else:
            out.append(vals[i])
            i += 1
    return tuple(out)


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


def set_call_expr_arg(var, args, kws, arg_no, arg_name):
    """replaces call argument with a new variable.
    Raises an error if argument was not specified.
    """
    if len(args) > arg_no:
        args[arg_no] = var
    elif arg_name in kws:
        kws[arg_name] = var
    else:
        raise BodoError(
            "cannot set call argument since does not exist"
        )  # pragma: no cover


def replace_func(
    pass_info,
    func,
    args,
    const=False,
    pre_nodes=None,
    extra_globals=None,
    pysig=None,
    kws=None,
):
    """
    """
    glbls = {"numba": numba, "np": np, "bodo": bodo, "pd": pd}
    if extra_globals is not None:
        glbls.update(extra_globals)
    func.__globals__.update(glbls)

    # create explicit arg variables for defaults if func has any
    # XXX: inine_closure_call() can't handle defaults properly
    if pysig is not None:
        pre_nodes = [] if pre_nodes is None else pre_nodes
        scope = next(iter(pass_info.func_ir.blocks.values())).scope
        loc = scope.loc

        def normal_handler(index, param, default):
            return default

        def default_handler(index, param, default):
            d_var = ir.Var(scope, mk_unique_var("defaults"), loc)
            pass_info.typemap[d_var.name] = numba.typeof(default)
            node = ir.Assign(ir.Const(default, loc), d_var, loc)
            pre_nodes.append(node)
            return d_var

        # TODO: stararg needs special handling?
        args = numba.core.typing.fold_arguments(
            pysig, args, kws, normal_handler, default_handler, normal_handler
        )

    arg_typs = tuple(pass_info.typemap[v.name] for v in args)

    if const:
        new_args = []
        for i, arg in enumerate(args):
            val = guard(find_const, pass_info.func_ir, arg)
            if val:
                new_args.append(types.literal(val))
            else:
                new_args.append(arg_typs[i])
        arg_typs = tuple(new_args)
    return ReplaceFunc(func, arg_typs, args, glbls, pre_nodes)


############################# UDF utils ############################


def is_var_size_item_array_type(t):
    """returns True if array type 't' has variable size items (e.g. strings)
    """
    assert is_array_typ(t, False)
    return (
        t == string_array_type
        or isinstance(t, ArrayItemArrayType)
        or (
            isinstance(t, StructArrayType)
            and any(is_var_size_item_array_type(d) for d in t.data)
        )
    )


def gen_init_varsize_alloc_sizes(t):
    """generate initialization code as text for allocation sizes for arrays with
    variable items, e.g. total number of characters in string arrays
    """
    # TODO: handle all possible array types and nested cases, e.g. struct
    if t == string_array_type:
        vname = "num_chars_{}".format(ir_utils.next_label())
        return f"  {vname} = 0\n", (vname,)
    if isinstance(t, ArrayItemArrayType):
        inner_code, inner_vars = gen_init_varsize_alloc_sizes(t.dtype)
        vname = "num_items_{}".format(ir_utils.next_label())
        return f"  {vname} = 0\n" + inner_code, (vname,) + inner_vars
    return "", ()


def gen_varsize_item_sizes(t, item, var_names):
    """generate aggregation code as text for allocation sizes for arrays with
    variable items, e.g. total number of characters in string arrays
    """
    # TODO: handle all possible array types and nested cases, e.g. struct
    if t == string_array_type:
        return "    {} += bodo.libs.str_arr_ext.get_utf8_size({})\n".format(
            var_names[0], item
        )
    if isinstance(t, ArrayItemArrayType):
        return "    {} += len({})\n".format(
            var_names[0], item
        ) + gen_varsize_array_counts(t.dtype, item, var_names[1:])
    return ""


def gen_varsize_array_counts(t, item, var_names):
    """count the total number of elements in a nested array. e.g. total characters in a
    string array.
    """
    # TODO: other arrays
    if t == string_array_type:
        return "    {} += bodo.libs.str_arr_ext.get_num_total_chars({})\n".format(
            var_names[0], item
        )


def get_type_alloc_counts(t):
    """get the number of counts needed for upfront allocation of array of type 't'.
    For example, ArrayItemArrayType(ArrayItemArrayType(array(int64))) returns 3.
    """
    if isinstance(t, StructArrayType):
        return 1 + sum(get_type_alloc_counts(d.dtype) for d in t.data)

    if isinstance(t, ArrayItemArrayType) or t == string_array_type:
        return 1 + get_type_alloc_counts(t.dtype)

    if bodo.utils.utils.is_array_typ(t, False) or t == bodo.string_type:
        return 1

    if isinstance(t, StructType):
        return sum(get_type_alloc_counts(d) for d in t.data)

    return 0
