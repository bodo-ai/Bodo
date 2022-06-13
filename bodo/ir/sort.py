# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""IR node for the data sorting"""
from collections import defaultdict
from typing import Dict, List, Union

import numba
import numpy as np
from numba.core import ir, ir_utils, typeinfer, types
from numba.core.ir_utils import (
    compile_to_numba_ir,
    mk_unique_var,
    replace_arg_nodes,
    replace_vars_inner,
    visit_vars_inner,
)

import bodo
import bodo.libs.timsort
from bodo.libs.array import (
    arr_info_list_to_table,
    array_to_info,
    delete_table,
    delete_table_decref_arrays,
    info_from_table,
    info_to_array,
    sort_values_table,
)
from bodo.libs.str_arr_ext import cp_str_list_to_array, to_list_if_immutable_arr
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.transforms.distributed_analysis import Distribution
from bodo.utils.utils import gen_getitem


class Sort(ir.Stmt):
    def __init__(
        self,
        df_in: str,
        df_out: str,
        key_arrs: List[ir.Var],
        out_key_arrs: List[ir.Var],
        df_in_vars: Dict[str, ir.Var],
        df_out_vars: Dict[str, ir.Var],
        inplace: bool,
        loc: ir.Loc,
        ascending_list: Union[List[bool], bool] = True,
        na_position: Union[List[str], str] = "last",
    ):
        """IR node for sort operations. Produced by several sort operations like
        df/Series.sort_values and df/Series.sort_index.
        Sort IR node allows analysis and optimization throughout the compiler pipeline
        such as removing dead columns. The implementation calls Timsort in the C++
        runtime.

        Args:
            df_in (str): name of input dataframe (for printing Sort node only)
            df_out (str): name of output dataframe (for printing Sort node only)
            key_arrs (list[ir.Var]): key arrays for sorting
            out_key_arrs (list[ir.Var]): output key arrays after sorting
            df_in_vars (dict[str, ir.Var]): dict mapping column names to input variables
            df_out_vars (dict[str, ir.Var]): dict mapping column names to output
                variables
            inplace (bool): sort values inplace (avoid creating new arrays)
            loc (ir.Loc): location object of this IR node
            ascending_list (bool|list[bool], optional): Ascending or descending sort
                order (can be set per key). Defaults to True.
            na_position (str|list[str], optional): Place null values first or last in
                output array. Can be set per key. Defaults to "last".
        """

        self.df_in = df_in
        self.df_out = df_out
        self.key_arrs = key_arrs
        self.out_key_arrs = out_key_arrs
        self.df_in_vars = df_in_vars
        self.df_out_vars = df_out_vars
        self.inplace = inplace

        # normalize na_position to list of bools (per key)
        if isinstance(na_position, str):
            if na_position == "last":
                self.na_position_b = (True,) * len(key_arrs)
            else:
                self.na_position_b = (False,) * len(key_arrs)
        else:
            self.na_position_b = tuple(
                [
                    True if col_na_position == "last" else False
                    for col_na_position in na_position
                ]
            )

        # normalize ascending to list of bools (per key)
        if isinstance(ascending_list, bool):
            ascending_list = (ascending_list,) * len(key_arrs)

        self.ascending_list = ascending_list
        self.loc = loc

    def __repr__(self):  # pragma: no cover
        in_cols = ""
        for (c, v) in self.df_in_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_in_str = "{}{{{}}}".format(self.df_in, in_cols)
        out_cols = ""
        for (c, v) in self.df_out_vars.items():
            out_cols += "'{}':{}, ".format(c, v.name)
        df_out_str = "{}{{{}}}".format(self.df_out, out_cols)
        return "sort: [key: {}] {} [key: {}] {}".format(
            ", ".join(v.name for v in self.key_arrs),
            df_in_str,
            ", ".join(v.name for v in self.out_key_arrs),
            df_out_str,
        )


def sort_array_analysis(sort_node, equiv_set, typemap, array_analysis):
    """Array analysis for Sort IR node. Input arrays have the same size. Output arrays
    have the same size as well.
    Inputs and outputs may not have the same local size after shuffling in parallel sort
    so we avoid adding equivalence for them to be conservative (1D_Var handling is
    challenging so the gains don't seem worth it).

    Args:
        sort_node (ir.Sort): input Sort node
        equiv_set (SymbolicEquivSet): equivalence set object of Numba array analysis
        typemap (dict[str, types.Type]): typemap from analysis pass
        array_analysis (ArrayAnalysis): array analysis object for the pass

    Returns:
        tuple(list(ir.Stmt), list(ir.Stmt)): lists of IR statements to add to IR before
        this node and after this node.
    """

    # arrays of input df have same size in first dimension
    all_shapes = []
    in_arrs = sort_node.key_arrs + list(sort_node.df_in_vars.values())
    for col_var in in_arrs:
        col_shape = equiv_set.get_shape(col_var)
        if col_shape is not None:
            all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variables for output columns
    post = []
    all_shapes = []
    out_vars = sort_node.out_key_arrs + list(sort_node.df_out_vars.values())

    for col_var in out_vars:
        typ = typemap[col_var.name]
        shape = array_analysis._gen_shape_call(equiv_set, col_var, typ.ndim, None, post)
        equiv_set.insert_equiv(col_var, shape)
        all_shapes.append(shape[0])
        equiv_set.define(col_var, set())

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post


numba.parfors.array_analysis.array_analysis_extensions[Sort] = sort_array_analysis


def sort_distributed_analysis(sort_node, array_dists):
    """Distributed analysis for Sort IR node. Inputs and outputs have the same
    distribution, except that output of 1D is 1D_Var due to shuffling.

    Args:
        sort_node (Sort): Sort IR node
        array_dists (dict[str, Distribution]): distributions of arrays in the IR
            (variable name -> Distribution)
    """

    in_arrs = sort_node.key_arrs + list(sort_node.df_in_vars.values())
    out_arrs = sort_node.out_key_arrs + list(sort_node.df_out_vars.values())
    # input columns have same distribution
    in_dist = Distribution.OneD
    for col_var in in_arrs:
        in_dist = Distribution(min(in_dist.value, array_dists[col_var.name].value))

    # output is 1D_Var due to shuffle, has to meet input dist
    # TODO: set to input dist in inplace case
    out_dist = Distribution(min(in_dist.value, Distribution.OneD_Var.value))
    for col_var in out_arrs:
        if col_var.name in array_dists:
            out_dist = Distribution(
                min(out_dist.value, array_dists[col_var.name].value)
            )

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        in_dist = out_dist

    # set dists
    for col_var in in_arrs:
        array_dists[col_var.name] = in_dist

    for col_var in out_arrs:
        array_dists[col_var.name] = out_dist


distributed_analysis.distributed_analysis_extensions[Sort] = sort_distributed_analysis


def sort_typeinfer(sort_node, typeinferer):
    """Type inference extension for Sort IR nodes. Corresponding input and output
    variables have the same type.

    Args:
        sort_node (Sort): Sort IR node
        typeinferer (TypeInferer): type inference pass object
    """

    # input and output arrays have the same type
    for in_key, out_key in zip(sort_node.key_arrs, sort_node.out_key_arrs):
        typeinferer.constraints.append(
            typeinfer.Propagate(dst=out_key.name, src=in_key.name, loc=sort_node.loc)
        )
    for col_name, col_var in sort_node.df_in_vars.items():
        out_col_var = sort_node.df_out_vars[col_name]
        typeinferer.constraints.append(
            typeinfer.Propagate(
                dst=out_col_var.name, src=col_var.name, loc=sort_node.loc
            )
        )


typeinfer.typeinfer_extensions[Sort] = sort_typeinfer


def build_sort_definitions(sort_node, definitions=None):
    """Sort IR node extension for building varibale definitions pass

    Args:
        sort_node (Sort): Sort IR node
        definitions (defaultdict(list), optional): Existing definitions list. Defaults
            to None.

    Returns:
        defaultdict(list): updated definitions
    """

    if definitions is None:
        definitions = defaultdict(list)

    # output arrays are defined
    if not sort_node.inplace:
        for col_var in sort_node.out_key_arrs + list(sort_node.df_out_vars.values()):
            definitions[col_var.name].append(sort_node)

    return definitions


ir_utils.build_defs_extensions[Sort] = build_sort_definitions


def visit_vars_sort(sort_node, callback, cbdata):
    """Sort IR node extension for visiting variables pass

    Args:
        sort_node (Sort): Sort IR node
        callback (function): callback to call on each variable (just passed along here)
        cbdata (object): data to pass to callback (just passed along here)
    """

    for i in range(len(sort_node.key_arrs)):
        sort_node.key_arrs[i] = visit_vars_inner(
            sort_node.key_arrs[i], callback, cbdata
        )
        sort_node.out_key_arrs[i] = visit_vars_inner(
            sort_node.out_key_arrs[i], callback, cbdata
        )

    for col_name in list(sort_node.df_in_vars.keys()):
        sort_node.df_in_vars[col_name] = visit_vars_inner(
            sort_node.df_in_vars[col_name], callback, cbdata
        )

    for col_name in list(sort_node.df_out_vars.keys()):
        sort_node.df_out_vars[col_name] = visit_vars_inner(
            sort_node.df_out_vars[col_name], callback, cbdata
        )


ir_utils.visit_vars_extensions[Sort] = visit_vars_sort


def remove_dead_sort(
    sort_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):
    """Dead code elimination for Sort IR node

    Args:
        sort_node (Sort): Sort IR node
        lives_no_aliases (set(str)): live variable names without their aliases
        lives (set(str)): live variable names with their aliases
        arg_aliases (set(str)): variables that are function arguments or alias them
        alias_map (dict(str, set(str))): mapping of variables names and their aliases
        func_ir (FunctionIR): full function IR
        typemap (dict(str, types.Type)): typemap of variables

    Returns:
        (Sort, optional): Sort IR node if not fully dead, None otherwise
    """

    # TODO: arg aliases for inplace case?
    dead_cols = []

    for col_name, col_var in sort_node.df_out_vars.items():
        if col_var.name not in lives:
            dead_cols.append(col_name)

    for cname in dead_cols:
        sort_node.df_in_vars.pop(cname)
        sort_node.df_out_vars.pop(cname)

    # remove empty sort node
    if len(sort_node.df_out_vars) == 0 and all(
        v.name not in lives for v in sort_node.out_key_arrs
    ):
        return None

    return sort_node


ir_utils.remove_dead_extensions[Sort] = remove_dead_sort


def sort_usedefs(sort_node, use_set=None, def_set=None):
    """use/def analysis extension for Sort IR node

    Args:
        sort_node (Sort): Sort IR node
        use_set (set(str), optional): Existing set of used variables. Defaults to None.
        def_set (set(str), optional): Existing set of defined variables. Defaults to
            None.

    Returns:
        namedtuple('use_defs_result', 'usemap,defmap'): use/def sets
    """
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # key array and input columns are used
    use_set.update({v.name for v in sort_node.key_arrs})
    use_set.update({v.name for v in sort_node.df_in_vars.values()})

    # output arrays are defined
    if not sort_node.inplace:
        def_set.update({v.name for v in sort_node.out_key_arrs})
        def_set.update({v.name for v in sort_node.df_out_vars.values()})

    return numba.core.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.core.analysis.ir_extension_usedefs[Sort] = sort_usedefs


def get_copies_sort(sort_node, typemap):
    """Sort IR node extension for variable copy analysis

    Args:
        sort_node (Sort): Sort IR node
        typemap (dict(str, ir.Var)): typemap of variables

    Returns:
        tuple(set(str), set(str)): set of copies generated or killed
    """
    # sort doesn't generate copies, it just kills the output columns
    kill_set = set()
    if not sort_node.inplace:
        kill_set = set(v.name for v in sort_node.df_out_vars.values())
        kill_set.update({v.name for v in sort_node.out_key_arrs})
    return set(), kill_set


ir_utils.copy_propagate_extensions[Sort] = get_copies_sort


def apply_copies_sort(
    sort_node, var_dict, name_var_table, typemap, calltypes, save_copies
):
    """Sort IR node extension for applying variable copies pass

    Args:
        sort_node (Sort): Sort IR node
        var_dict (dict(str, ir.Var)): dictionary of variables to replace
        name_var_table (dict(str, ir.Var)): map variable name to its ir.Var object
        typemap (dict(str, ir.Var)): typemap of variables
        calltypes (dict[ir.Inst, Signature]): signature of callable nodes
        save_copies (list(tuple(str, ir.Var))): copies that were applied
    """
    for i in range(len(sort_node.key_arrs)):
        sort_node.key_arrs[i] = replace_vars_inner(sort_node.key_arrs[i], var_dict)
        sort_node.out_key_arrs[i] = replace_vars_inner(
            sort_node.out_key_arrs[i], var_dict
        )

    for col_name in list(sort_node.df_in_vars.keys()):
        sort_node.df_in_vars[col_name] = replace_vars_inner(
            sort_node.df_in_vars[col_name], var_dict
        )

    for col_name in list(sort_node.df_out_vars.keys()):
        sort_node.df_out_vars[col_name] = replace_vars_inner(
            sort_node.df_out_vars[col_name], var_dict
        )


ir_utils.apply_copy_propagate_extensions[Sort] = apply_copies_sort


def sort_distributed_run(
    sort_node, array_dists, typemap, calltypes, typingctx, targetctx
):
    """lowers Sort IR node to regular IR nodes. Uses the C++ Timsort implementation

    Args:
        sort_node (Sort): Sort IR node to lower
        array_dists (dict(str, Distribution)): distribution of arrays
        typemap (dict(str, ir.Var)): typemap of variables
        calltypes (dict[ir.Inst, Signature]): signature of callable nodes
        typingctx (typing.Context): typing context for compiler pipeline
        targetctx (cpu.CPUContext): target context for compiler pipeline

    Returns:
        list(ir.Stmt): list of IR nodes that implement the input Sort IR node
    """

    parallel = False
    in_vars = list(sort_node.df_in_vars.values())
    out_vars = list(sort_node.df_out_vars.values())
    if array_dists is not None:
        parallel = True
        for v in sort_node.key_arrs + sort_node.out_key_arrs + in_vars + out_vars:
            if (
                array_dists[v.name] != distributed_pass.Distribution.OneD
                and array_dists[v.name] != distributed_pass.Distribution.OneD_Var
            ):
                parallel = False

    loc = sort_node.loc
    scope = sort_node.key_arrs[0].scope
    # copy arrays when not inplace
    nodes = []
    key_arrs = sort_node.key_arrs
    if not sort_node.inplace:
        new_keys = []
        for v in key_arrs:
            new_key = _copy_array_nodes(
                v, nodes, typingctx, targetctx, typemap, calltypes
            )
            new_keys.append(new_key)
        key_arrs = new_keys
        new_in_vars = []
        for v in in_vars:
            v_cp = _copy_array_nodes(v, nodes, typingctx, targetctx, typemap, calltypes)
            new_in_vars.append(v_cp)
        in_vars = new_in_vars

    key_name_args = [f"key{i}" for i in range(len(key_arrs))]
    key_name_args_join = ", ".join(key_name_args)
    col_name_args = [f"c{i}" for i in range(len(in_vars))]
    col_name_args_join = ", ".join(col_name_args)

    func_text = f"def f({key_name_args_join}, {col_name_args_join}):\n"
    func_text += get_sort_cpp_section(
        key_name_args,
        col_name_args,
        sort_node.ascending_list,
        sort_node.na_position_b,
        parallel,
    )
    func_text += "  return key_arrs, data\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    sort_impl = loc_vars["f"]

    key_typ = types.Tuple([typemap[v.name] for v in key_arrs])
    data_tup_typ = types.Tuple([typemap[v.name] for v in in_vars])

    f_block = compile_to_numba_ir(
        sort_impl,
        {
            "bodo": bodo,
            "np": np,
            "to_list_if_immutable_arr": to_list_if_immutable_arr,
            "cp_str_list_to_array": cp_str_list_to_array,
            "delete_table": delete_table,
            "delete_table_decref_arrays": delete_table_decref_arrays,
            "info_to_array": info_to_array,
            "info_from_table": info_from_table,
            "sort_values_table": sort_values_table,
            "arr_info_list_to_table": arr_info_list_to_table,
            "array_to_info": array_to_info,
        },
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=tuple(list(key_typ.types) + list(data_tup_typ.types)),
        typemap=typemap,
        calltypes=calltypes,
    ).blocks.popitem()[1]
    replace_arg_nodes(f_block, key_arrs + in_vars)
    nodes += f_block.body[:-2]
    ret_var = nodes[-1].target
    # get key tup
    key_arrs_tup_var = ir.Var(scope, mk_unique_var("key_data"), loc)
    typemap[key_arrs_tup_var.name] = key_typ
    gen_getitem(key_arrs_tup_var, ret_var, 0, calltypes, nodes)
    # get data tup
    data_tup_var = ir.Var(scope, mk_unique_var("sort_data"), loc)
    typemap[data_tup_var.name] = data_tup_typ
    gen_getitem(data_tup_var, ret_var, 1, calltypes, nodes)

    for i, var in enumerate(sort_node.out_key_arrs):
        gen_getitem(var, key_arrs_tup_var, i, calltypes, nodes)
    for i, var in enumerate(out_vars):
        gen_getitem(var, data_tup_var, i, calltypes, nodes)
    # TODO: handle 1D balance for inplace case
    return nodes


distributed_pass.distributed_run_extensions[Sort] = sort_distributed_run


def _copy_array_nodes(var, nodes, typingctx, targetctx, typemap, calltypes):
    """generate IR nodes for copying an array

    Args:
        var (ir.Var): variable of array to copy
        nodes (list[ir.Stmt]): list of IR nodes to add output to
        typingctx (typing.Context): typing context for compiler pipeline
        targetctx (cpu.CPUContext): target context for compiler pipeline
        typemap (dict(str, ir.Var)): typemap of variables
        calltypes (dict[ir.Inst, Signature]): signature of callable nodes
    """

    def _impl(arr):  # pragma: no cover
        return arr.copy()

    f_block = compile_to_numba_ir(
        _impl,
        {},
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=(typemap[var.name],),
        typemap=typemap,
        calltypes=calltypes,
    ).blocks.popitem()[1]
    replace_arg_nodes(f_block, [var])
    nodes += f_block.body[:-2]
    return nodes[-1].target


def get_sort_cpp_section(
    key_name_args, col_name_args, ascending_list, na_position_b, parallel
):
    """generate function text for pass arrays to C++ and calling sort.

    Args:
        key_name_args (list(str)): list of argument names for key inputs
        col_name_args (list(str)): list of argument names for data inputs
        ascending_list (list(bool)): list of ascending/descending order for each key
        na_position_b (list(bool)): list of first/list null position for each key
        parallel (bool): flag for parallel sort

    Returns:
        str: function text for calling C++ sort.
    """
    func_text = ""
    key_count = len(key_name_args)
    total_list = [f"array_to_info({name})" for name in key_name_args] + [
        f"array_to_info({name})" for name in col_name_args
    ]
    func_text += "  info_list_total = [{}]\n".format(",".join(total_list))
    func_text += "  table_total = arr_info_list_to_table(info_list_total)\n"
    func_text += "  vect_ascending = np.array([{}], np.int64)\n".format(
        ",".join("1" if x else "0" for x in ascending_list)
    )
    func_text += "  na_position = np.array([{}], np.int64)\n".format(
        ",".join("1" if x else "0" for x in na_position_b)
    )
    func_text += f"  out_table = sort_values_table(table_total, {key_count}, vect_ascending.ctypes, na_position.ctypes, {parallel})\n"
    idx = 0
    list_key_str = []
    for name in key_name_args:
        list_key_str.append(f"info_to_array(info_from_table(out_table, {idx}), {name})")
        idx += 1
    func_text += "  key_arrs = ({},)\n".format(",".join(list_key_str))

    list_data_str = []
    for name in col_name_args:
        list_data_str.append(
            f"info_to_array(info_from_table(out_table, {idx}), {name})"
        )
        idx += 1
    if len(list_data_str) > 0:
        func_text += "  data = ({},)\n".format(",".join(list_data_str))
    else:
        func_text += "  data = ()\n"
    func_text += "  delete_table(out_table)\n"
    func_text += "  delete_table(table_total)\n"
    return func_text
