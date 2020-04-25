# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""IR node for the data sorting"""
import numpy as np
import math
from collections import defaultdict
import numba
from numba.core import ir, ir_utils, typeinfer, types
from numba.core.ir_utils import (
    visit_vars_inner,
    replace_vars_inner,
    compile_to_numba_ir,
    replace_arg_nodes,
    mk_unique_var,
)
from bodo.libs.array import (
    array_to_info,
    arr_info_list_to_table,
    sort_values_table,
    info_from_table,
    info_to_array,
    delete_table,
)


import bodo
import bodo.libs.timsort
from bodo.transforms import distributed_pass, distributed_analysis
from bodo.libs.distributed_api import Reduce_Type
from bodo.transforms.distributed_analysis import Distribution
from bodo.utils.utils import debug_prints, empty_like_type, gen_getitem

from bodo.utils.shuffle import (
    alltoallv_tup,
    finalize_shuffle_meta,
    update_shuffle_meta,
    alloc_pre_shuffle_metadata,
    _get_keys_tup,
    _get_data_tup,
)

from bodo.libs.str_arr_ext import (
    string_array_type,
    to_string_list,
    cp_str_list_to_array,
    str_list_to_array,
    get_offset_ptr,
    get_data_ptr,
    pre_alloc_string_array,
    num_total_chars,
)
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.bool_arr_ext import boolean_array
from bodo.hiframes.datetime_date_ext import datetime_date_array_type


MIN_SAMPLES = 1000000
# MIN_SAMPLES = 100
samplePointsPerPartitionHint = 20
MPI_ROOT = 0


class Sort(ir.Stmt):
    def __init__(
        self,
        df_in,
        df_out,
        key_arrs,
        out_key_arrs,
        df_in_vars,
        df_out_vars,
        inplace,
        loc,
        ascending_list=True,
        na_position="last",
    ):
        # for printing only
        self.df_in = df_in
        self.df_out = df_out
        self.key_arrs = key_arrs
        self.out_key_arrs = out_key_arrs
        self.df_in_vars = df_in_vars
        self.df_out_vars = df_out_vars
        self.inplace = inplace
        if na_position == "last":
            self.na_position_b = True
        else:
            self.na_position_b = False
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
        (shape, c_post) = array_analysis._gen_shape_call(
            equiv_set, col_var, typ.ndim, None
        )
        equiv_set.insert_equiv(col_var, shape)
        post.extend(c_post)
        all_shapes.append(shape[0])
        equiv_set.define(col_var, set())

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post


numba.parfors.array_analysis.array_analysis_extensions[Sort] = sort_array_analysis


def sort_distributed_analysis(sort_node, array_dists):

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

    # TODO: handle rebalance
    # assert not (in_dist == Distribution.OneD and out_dist == Distribution.OneD_Var)
    return


distributed_analysis.distributed_analysis_extensions[Sort] = sort_distributed_analysis


def sort_typeinfer(sort_node, typeinferer):
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
    return


typeinfer.typeinfer_extensions[Sort] = sort_typeinfer


def build_sort_definitions(sort_node, definitions=None):
    if definitions is None:
        definitions = defaultdict(list)

    # output arrays are defined
    if not sort_node.inplace:
        for col_var in sort_node.out_key_arrs + list(sort_node.df_out_vars.values()):
            definitions[col_var.name].append(sort_node)

    return definitions


ir_utils.build_defs_extensions[Sort] = build_sort_definitions


def visit_vars_sort(sort_node, callback, cbdata):
    if debug_prints():  # pragma: no cover
        print("visiting sort vars for:", sort_node)
        print("cbdata: ", sorted(cbdata.items()))

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


# add call to visit sort variable
ir_utils.visit_vars_extensions[Sort] = visit_vars_sort


def remove_dead_sort(
    sort_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):

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
    """apply copy propagate in sort node"""
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

    return


ir_utils.apply_copy_propagate_extensions[Sort] = apply_copies_sort


def sort_distributed_run(
    sort_node, array_dists, typemap, calltypes, typingctx, targetctx, dist_pass
):
    parallel = True
    in_vars = list(sort_node.df_in_vars.values())
    out_vars = list(sort_node.df_out_vars.values())
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
            new_key = _copy_array_nodes(v, nodes, typingctx, typemap, calltypes)
            new_keys.append(new_key)
        key_arrs = new_keys
        new_in_vars = []
        for v in in_vars:
            v_cp = _copy_array_nodes(v, nodes, typingctx, typemap, calltypes)
            new_in_vars.append(v_cp)
        in_vars = new_in_vars

    key_name_args = ["key" + str(i) for i in range(len(key_arrs))]
    key_name_args_join = ", ".join(key_name_args)
    col_name_args = ["c" + str(i) for i in range(len(in_vars))]
    col_name_args_join = ", ".join(col_name_args)
    # TODO: use *args
    func_text = "def f({}, {}):\n".format(key_name_args_join, col_name_args_join)
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
            "to_string_list": to_string_list,
            "cp_str_list_to_array": cp_str_list_to_array,
            "delete_table": delete_table,
            "info_to_array": info_to_array,
            "info_from_table": info_from_table,
            "sort_values_table": sort_values_table,
            "arr_info_list_to_table": arr_info_list_to_table,
            "array_to_info": array_to_info,
        },
        typingctx,
        tuple(list(key_typ.types) + list(data_tup_typ.types)),
        typemap,
        calltypes,
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


def _copy_array_nodes(var, nodes, typingctx, typemap, calltypes):
    def _impl(arr):  # pragma: no cover
        return arr.copy()

    f_block = compile_to_numba_ir(
        _impl, {}, typingctx, (typemap[var.name],), typemap, calltypes
    ).blocks.popitem()[1]
    replace_arg_nodes(f_block, [var])
    nodes += f_block.body[:-2]
    return nodes[-1].target


def get_sort_cpp_section(key_name_args, col_name_args, ascending_list, na_position_b, parallel_b):
    key_count = len(key_name_args)
    data_count = len(col_name_args)
    total_list = ["array_to_info({})".format(name) for name in key_name_args] + [
        "array_to_info({})".format(name) for name in col_name_args
    ]
    func_text  = "  info_list_total = [{}]\n".format(",".join(total_list))
    func_text += "  table_total = arr_info_list_to_table(info_list_total)\n"
    func_text += "  vect_ascending = np.array([{}])\n".format(
        ",".join("1" if x else "0" for x in ascending_list)
    )
    func_text += "  out_table = sort_values_table(table_total, {}, vect_ascending.ctypes, {}, {})\n".format(
        key_count, na_position_b, parallel_b
    )
    idx = 0
    list_key_str = []
    for name in key_name_args:
        list_key_str.append(
            "info_to_array(info_from_table(out_table, {}), {})".format(idx, name)
        )
        idx += 1
    func_text += "  key_arrs = ({},)\n".format(",".join(list_key_str))

    list_data_str = []
    for name in col_name_args:
        list_data_str.append(
            "info_to_array(info_from_table(out_table, {}), {})".format(idx, name)
        )
        idx += 1
    if len(list_data_str) > 0:
        func_text += "  data = ({},)\n".format(",".join(list_data_str))
    else:
        func_text += "  data = ()\n"
    func_text += "  delete_table(out_table)\n"
    func_text += "  delete_table(table_total)\n"
    return func_text
