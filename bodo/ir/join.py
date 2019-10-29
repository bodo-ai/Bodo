# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
from collections import defaultdict
import numpy as np
import pandas as pd

import numba
from numba import typeinfer, ir, ir_utils, config, types, generated_jit
from numba.extending import overload
from numba.ir_utils import (
    visit_vars_inner,
    replace_vars_inner,
    compile_to_numba_ir,
    replace_arg_nodes,
    mk_unique_var,
)
import bodo
from bodo import objmode
from bodo.transforms import distributed_pass, distributed_analysis
from bodo.utils.utils import debug_prints, alloc_arr_tup
from bodo.transforms.distributed_analysis import Distribution

from bodo.libs.str_arr_ext import (
    string_array_type,
    to_string_list,
    cp_str_list_to_array,
    get_bit_bitmap,
    num_total_chars,
    get_offset_ptr,
    get_data_ptr,
    get_null_bitmap_ptr,
    pre_alloc_string_array,
    getitem_str_offset,
    copy_str_arr_slice,
    str_copy_ptr,
    setitem_str_offset,
    str_arr_set_na,
    set_bit_to,
    print_str_arr,
    get_str_arr_item_ptr,
    get_str_arr_item_length,
    get_utf8_size,
)
from bodo.libs.str_ext import string_type
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.timsort import copyElement_tup, getitem_arr_tup, setitem_arr_tup
from bodo.utils.shuffle import (
    getitem_arr_tup_single,
    val_to_tup,
    alltoallv_tup,
    finalize_shuffle_meta,
    update_shuffle_meta,
    alloc_pre_shuffle_metadata,
    _get_keys_tup,
    _get_data_tup,
)
from bodo.libs.array_tools import (
    array_to_info,
    arr_info_list_to_table,
    shuffle_table,
    hash_join_table,
    info_from_table,
    info_to_array,
    delete_table,
)
from bodo.hiframes.pd_categorical_ext import CategoricalArray


class Join(ir.Stmt):
    def __init__(
        self,
        df_out,
        left_df,
        right_df,
        left_keys,
        right_keys,
        out_vars,
        left_vars,
        right_vars,
        how,
        loc,
    ):
        self.df_out = df_out
        self.left_df = left_df
        self.right_df = right_df
        self.left_keys = left_keys
        self.right_keys = right_keys
        self.df_out_vars = out_vars
        self.left_vars = left_vars
        self.right_vars = right_vars
        self.how = how
        self.loc = loc

        # keep the origin of output columns to enable proper dead code elimination
        # columns with common name that are not common keys will get a suffix
        comm_keys = set(left_keys) & set(right_keys)
        comm_data = set(left_vars.keys()) & set(right_vars.keys())
        add_suffix = comm_data - comm_keys

        self.column_origins = {
            (c + "_x" if c in add_suffix else c): ("left", c) for c in left_vars.keys()
        }
        self.column_origins.update(
            {
                (c + "_y" if c in add_suffix else c): ("right", c)
                for c in right_vars.keys()
            }
        )

    def __repr__(self):  # pragma: no cover
        out_cols = ""
        for (c, v) in self.df_out_vars.items():
            out_cols += "'{}':{}, ".format(c, v.name)
        df_out_str = "{}{{{}}}".format(self.df_out, out_cols)

        in_cols = ""
        for (c, v) in self.left_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_left_str = "{}{{{}}}".format(self.left_df, in_cols)

        in_cols = ""
        for (c, v) in self.right_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_right_str = "{}{{{}}}".format(self.right_df, in_cols)
        return "join [{}={}]: {} , {}, {}".format(
            self.left_keys, self.right_keys, df_out_str, df_left_str, df_right_str
        )


def join_array_analysis(join_node, equiv_set, typemap, array_analysis):
    post = []
    # empty join nodes should be deleted in remove dead
    assert len(join_node.df_out_vars) > 0, "empty join in array analysis"

    # arrays of left_df and right_df have same size in first dimension
    all_shapes = []
    in_vars = list(join_node.left_vars.values())
    for col_var in in_vars:
        typ = typemap[col_var.name]
        col_shape = equiv_set.get_shape(col_var)
        all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    all_shapes = []
    in_vars = list(join_node.right_vars.values())
    for col_var in in_vars:
        typ = typemap[col_var.name]
        col_shape = equiv_set.get_shape(col_var)
        all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variable for an output column
    all_shapes = []
    for col_var in join_node.df_out_vars.values():
        typ = typemap[col_var.name]
        (shape, c_post) = array_analysis._gen_shape_call(
            equiv_set, col_var, typ.ndim, None
        )
        equiv_set.insert_equiv(col_var, shape)
        post.extend(c_post)
        all_shapes.append(shape[0])
        equiv_set.define(col_var)

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post


numba.array_analysis.array_analysis_extensions[Join] = join_array_analysis


def join_distributed_analysis(join_node, array_dists):

    # TODO: can columns of the same input table have diffrent dists?
    # left and right inputs can have 1D or 1D_Var seperately (q26 case)
    # input columns have same distribution
    left_dist = Distribution.OneD
    right_dist = Distribution.OneD
    for col_var in join_node.left_vars.values():
        left_dist = Distribution(min(left_dist.value, array_dists[col_var.name].value))

    for col_var in join_node.right_vars.values():
        right_dist = Distribution(
            min(right_dist.value, array_dists[col_var.name].value)
        )

    # output columns have same distribution
    out_dist = Distribution.OneD_Var
    for col_var in join_node.df_out_vars.values():
        # output dist might not be assigned yet
        if col_var.name in array_dists:
            out_dist = Distribution(
                min(out_dist.value, array_dists[col_var.name].value)
            )

    # out dist should meet input dist (e.g. REP in causes REP out)
    # output can be stay parallel if any of the inputs is parallel, hence max()
    out_dist1 = Distribution(min(out_dist.value, left_dist.value))
    out_dist2 = Distribution(min(out_dist.value, right_dist.value))
    out_dist = Distribution(max(out_dist1.value, out_dist2.value))
    for col_var in join_node.df_out_vars.values():
        array_dists[col_var.name] = out_dist

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        left_dist = out_dist
        right_dist = out_dist

    # assign input distributions
    for col_var in join_node.left_vars.values():
        array_dists[col_var.name] = left_dist

    for col_var in join_node.right_vars.values():
        array_dists[col_var.name] = right_dist

    return


distributed_analysis.distributed_analysis_extensions[Join] = join_distributed_analysis


def join_typeinfer(join_node, typeinferer):

    for out_col_name, out_col_var in join_node.df_out_vars.items():
        # left suffix
        if out_col_name.endswith("_x"):
            col_var = join_node.left_vars[out_col_name[:-2]]
        # right suffix
        elif out_col_name.endswith("_y"):
            col_var = join_node.right_vars[out_col_name[:-2]]
        elif out_col_name in join_node.left_vars:
            col_var = join_node.left_vars[out_col_name]
        else:
            assert out_col_name in join_node.right_vars
            col_var = join_node.right_vars[out_col_name]
        typeinferer.constraints.append(
            typeinfer.Propagate(
                dst=out_col_var.name, src=col_var.name, loc=join_node.loc
            )
        )
    return


typeinfer.typeinfer_extensions[Join] = join_typeinfer


def visit_vars_join(join_node, callback, cbdata):
    if debug_prints():  # pragma: no cover
        print("visiting join vars for:", join_node)
        print("cbdata: ", sorted(cbdata.items()))

    # left
    for col_name in list(join_node.left_vars.keys()):
        join_node.left_vars[col_name] = visit_vars_inner(
            join_node.left_vars[col_name], callback, cbdata
        )
    # right
    for col_name in list(join_node.right_vars.keys()):
        join_node.right_vars[col_name] = visit_vars_inner(
            join_node.right_vars[col_name], callback, cbdata
        )
    # output
    for col_name in list(join_node.df_out_vars.keys()):
        join_node.df_out_vars[col_name] = visit_vars_inner(
            join_node.df_out_vars[col_name], callback, cbdata
        )


# add call to visit Join variable
ir_utils.visit_vars_extensions[Join] = visit_vars_join


def remove_dead_join(join_node, lives, arg_aliases, alias_map, func_ir, typemap):
    # if an output column is dead, the related input column is not needed
    # anymore in the join
    dead_cols = []
    # TODO: remove output of dead keys
    all_cols_dead = True

    for col_name, col_var in join_node.df_out_vars.items():
        if col_var.name in lives:
            all_cols_dead = False
            continue
        orig, orig_name = join_node.column_origins[col_name]
        if orig == "left" and orig_name not in join_node.left_keys:
            join_node.left_vars.pop(orig_name)
            dead_cols.append(col_name)
        if orig == "right" and orig_name not in join_node.right_keys:
            join_node.right_vars.pop(orig_name)
            dead_cols.append(col_name)

    for cname in dead_cols:
        join_node.df_out_vars.pop(cname)

    # remove empty join node
    if all_cols_dead:
        return None

    return join_node


ir_utils.remove_dead_extensions[Join] = remove_dead_join


def join_usedefs(join_node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # input columns are used
    use_set.update({v.name for v in join_node.left_vars.values()})
    use_set.update({v.name for v in join_node.right_vars.values()})

    # output columns are defined
    def_set.update({v.name for v in join_node.df_out_vars.values()})

    return numba.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.analysis.ir_extension_usedefs[Join] = join_usedefs


def get_copies_join(join_node, typemap):
    # join doesn't generate copies, it just kills the output columns
    kill_set = set(v.name for v in join_node.df_out_vars.values())
    return set(), kill_set


ir_utils.copy_propagate_extensions[Join] = get_copies_join


def apply_copies_join(
    join_node, var_dict, name_var_table, typemap, calltypes, save_copies
):
    """apply copy propagate in join node"""

    # left
    for col_name in list(join_node.left_vars.keys()):
        join_node.left_vars[col_name] = replace_vars_inner(
            join_node.left_vars[col_name], var_dict
        )
    # right
    for col_name in list(join_node.right_vars.keys()):
        join_node.right_vars[col_name] = replace_vars_inner(
            join_node.right_vars[col_name], var_dict
        )
    # output
    for col_name in list(join_node.df_out_vars.keys()):
        join_node.df_out_vars[col_name] = replace_vars_inner(
            join_node.df_out_vars[col_name], var_dict
        )

    return


ir_utils.apply_copy_propagate_extensions[Join] = apply_copies_join


def build_join_definitions(join_node, definitions=None):
    if definitions is None:
        definitions = defaultdict(list)

    for col_var in join_node.df_out_vars.values():
        definitions[col_var.name].append(join_node)

    return definitions


ir_utils.build_defs_extensions[Join] = build_join_definitions


def join_distributed_run(
    join_node, array_dists, typemap, calltypes, typingctx, targetctx, dist_pass
):

    left_parallel, right_parallel = _get_table_parallel_flags(join_node, array_dists)

    method = "hash"
    pd_join_func = None
    if bodo.use_pandas_join:
        method = "pandas"
    # method = 'sort'
    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    loc = join_node.loc
    n_keys = len(join_node.left_keys)
    # get column variables
    left_key_vars = tuple(join_node.left_vars[c] for c in join_node.left_keys)
    right_key_vars = tuple(join_node.right_vars[c] for c in join_node.right_keys)

    left_other_col_vars = tuple(
        v
        for (n, v) in sorted(join_node.left_vars.items())
        if n not in join_node.left_keys
    )
    right_other_col_vars = tuple(
        v
        for (n, v) in sorted(join_node.right_vars.items())
        if n not in join_node.right_keys
    )
    # get column types
    arg_vars = (
        left_key_vars + right_key_vars + left_other_col_vars + right_other_col_vars
    )
    arg_typs = tuple(typemap[v.name] for v in arg_vars)
    scope = arg_vars[0].scope

    # arg names of non-key columns
    left_other_names = tuple("t1_c" + str(i) for i in range(len(left_other_col_vars)))
    right_other_names = tuple("t2_c" + str(i) for i in range(len(right_other_col_vars)))

    left_key_names = tuple("t1_key" + str(i) for i in range(n_keys))
    right_key_names = tuple("t2_key" + str(i) for i in range(n_keys))

    func_text = "def f({}, {},{}{}{}):\n".format(
        ",".join(left_key_names),
        ",".join(right_key_names),
        ",".join(left_other_names),
        ("," if len(left_other_names) != 0 else ""),
        ",".join(right_other_names),
    )

    left_key_types = tuple(typemap[v.name] for v in left_key_vars)
    right_key_types = tuple(typemap[v.name] for v in right_key_vars)

    func_text += "    t1_keys = ({},)\n".format(
        ", ".join(
            "{}{}".format(
                left_key_names[i],
                _gen_type_match(left_key_types[i], right_key_types[i]),
            )
            for i in range(n_keys)
        )
    )
    func_text += "    t2_keys = ({},)\n".format(
        ", ".join(
            "{}{}".format(
                right_key_names[i],
                _gen_type_match(right_key_types[i], left_key_types[i]),
            )
            for i in range(n_keys)
        )
    )

    func_text += "    data_left = ({}{})\n".format(
        ",".join(left_other_names), "," if len(left_other_names) != 0 else ""
    )
    func_text += "    data_right = ({}{})\n".format(
        ",".join(right_other_names), "," if len(right_other_names) != 0 else ""
    )

    if join_node.how == "asof":
        if left_parallel or right_parallel:
            assert left_parallel and right_parallel
            # only the right key needs to be aligned
            func_text += "    t2_keys, data_right = parallel_asof_comm(t1_keys, t2_keys, data_right)\n"
    else:
        if left_parallel:
            if bodo.use_legacy_shuffle:
                func_text += (
                    "    t1_keys, data_left = parallel_shuffle(t1_keys, data_left)\n"
                )
            else:
                func_text += _gen_par_shuffle(
                    left_key_names,
                    left_other_names,
                    "t1_keys",
                    "data_left",
                    left_key_types,
                    right_key_types,
                )
        if right_parallel:
            if bodo.use_legacy_shuffle:
                func_text += (
                    "    t2_keys, data_right = parallel_shuffle(t2_keys, data_right)\n"
                )
            else:
                func_text += _gen_par_shuffle(
                    right_key_names,
                    right_other_names,
                    "t2_keys",
                    "data_right",
                    right_key_types,
                    left_key_types,
                )
        # func_text += "    print(t2_key, data_right)\n"

    if method == "sort" and join_node.how != "asof":
        # asof key is already sorted, TODO: add error checking
        # local sort
        func_text += "    bodo.ir.sort.local_sort(t1_keys, data_left)\n"
        func_text += "    bodo.ir.sort.local_sort(t2_keys, data_right)\n"

    def _get_out_col_var(cname, is_left):
        if is_left and cname + "_x" in join_node.df_out_vars:
            return join_node.df_out_vars[cname + "_x"]
        if not is_left and cname + "_y" in join_node.df_out_vars:
            return join_node.df_out_vars[cname + "_y"]

        return join_node.df_out_vars[cname]

    # align output variables for local merge
    # add keys first (TODO: remove dead keys)
    out_l_key_vars = tuple(_get_out_col_var(c, True) for c in join_node.left_keys)
    out_r_key_vars = tuple(_get_out_col_var(c, False) for c in join_node.right_keys)
    # create dummy variable if right key is not actually returned
    # using the same output left key causes errors for asof case
    if join_node.left_keys == join_node.right_keys:
        out_r_key_vars = tuple(
            ir.Var(scope, mk_unique_var("dummy_k"), loc) for _ in range(n_keys)
        )
        for v, w in zip(out_r_key_vars, out_l_key_vars):
            typemap[v.name] = typemap[w.name]

    merge_out = out_l_key_vars + out_r_key_vars
    merge_out += tuple(
        _get_out_col_var(n, True)
        for (n, v) in sorted(join_node.left_vars.items())
        if n not in join_node.left_keys
    )
    merge_out += tuple(
        _get_out_col_var(n, False)
        for (n, v) in sorted(join_node.right_vars.items())
        if n not in join_node.right_keys
    )
    out_names = ["t3_c" + str(i) for i in range(len(merge_out))]

    if join_node.how == "asof":
        func_text += (
            "    out_t1_keys, out_t2_keys, out_data_left, out_data_right"
            " = bodo.ir.join.local_merge_asof(t1_keys, t2_keys, data_left, data_right)\n"
        )
    elif method == "sort":
        func_text += (
            "    out_t1_keys, out_t2_keys, out_data_left, out_data_right"
            " = bodo.ir.join.local_merge_new(t1_keys, t2_keys, data_left, data_right, {}, {})\n".format(
                join_node.how in ("left", "outer"), join_node.how == "outer"
            )
        )
    elif method == "pandas":
        pd_join_func = _gen_pd_join(
            left_key_vars,
            right_key_vars,
            left_other_col_vars,
            right_other_col_vars,
            typemap,
            join_node.how,
        )
        func_text += (
            "    out_t1_keys, out_t2_keys, out_data_left, out_data_right"
            " = pd_join_func(t1_keys, t2_keys, data_left, data_right, '{}', {})\n".format(
                join_node.how, join_node.left_keys == join_node.right_keys
            )
        )
    else:
        assert method == "hash"
        is_left = join_node.how in ("left", "outer")
        is_right = join_node.how == "outer"
        if bodo.use_cpp_hash_join:
            func_text += _gen_local_hash_join(left_key_names,right_key_names,left_other_names, right_other_names, is_left, is_right)
        else:
            func_text += (
                "    out_t1_keys, out_t2_keys, out_data_left, out_data_right"
                " = bodo.ir.join.local_hash_join(t1_keys, t2_keys, data_left, data_right, {}, {})\n".format(
                    is_left, is_right
                )
            )
    if bodo.use_cpp_hash_join == False or join_node.how == "asof" or method != "hash":
        for i in range(len(left_other_names)):
            func_text += "    left_{} = out_data_left[{}]\n".format(i, i)
        for i in range(len(right_other_names)):
            func_text += "    right_{} = out_data_right[{}]\n".format(i, i)
        for i in range(n_keys):
            func_text += "    t1_keys_{} = out_t1_keys[{}]{}\n".format(
                i, i, _gen_reverse_type_match(left_key_types[i], right_key_types[i])
            )
        for i in range(n_keys):
            func_text += "    t2_keys_{} = out_t2_keys[{}]\n".format(
                i, i, _gen_reverse_type_match(right_key_types[i], left_key_types[i])
            )

    for i in range(n_keys):
        func_text += "    {} = t1_keys_{}\n".format(out_names[i], i)
    for i in range(n_keys):
        func_text += "    {} = t2_keys_{}\n".format(out_names[n_keys + i], i)

    for i in range(len(left_other_names)):
        func_text += "    {} = left_{}\n".format(out_names[i + 2 * n_keys], i)

    for i in range(len(right_other_names)):
        func_text += "    {} = right_{}\n".format(
            out_names[i + 2 * n_keys + len(left_other_names)], i
        )

    loc_vars = {}
    print("func_text=", func_text)
    exec(func_text, {}, loc_vars)
    join_impl = loc_vars["f"]

    # print(func_text)

    glbs = {
        "bodo": bodo,
        "np": np,
        "pd_join_func": pd_join_func,
        "to_string_list": to_string_list,
        "cp_str_list_to_array": cp_str_list_to_array,
        "parallel_shuffle": parallel_shuffle,
        "parallel_asof_comm": parallel_asof_comm,
        "array_to_info": array_to_info,
        "arr_info_list_to_table": arr_info_list_to_table,
        "shuffle_table": shuffle_table,
        "hash_join_table": hash_join_table,
        "info_from_table": info_from_table,
        "info_to_array": info_to_array,
        "delete_table": delete_table,
    }

    f_block = compile_to_numba_ir(
        join_impl, glbs, typingctx, arg_typs, typemap, calltypes
    ).blocks.popitem()[1]
    replace_arg_nodes(f_block, arg_vars)

    nodes = f_block.body[:-3]
    for i in range(len(merge_out)):
        nodes[-len(merge_out) + i].target = merge_out[i]

    return nodes


distributed_pass.distributed_run_extensions[Join] = join_distributed_run


def _gen_type_match(t1, t2):
    """Match key types to have equal hash values.
    Python's hasher is consistent across types (e.g. int vs float) but others
    are not.
    """
    if (
        isinstance(t1, types.Array)
        and isinstance(t2, types.Array)
        and isinstance(t1.dtype, types.Float)
        and isinstance(t2.dtype, types.Integer)
    ):
        return ".astype(np.{})".format(t2.dtype)
    return ""


def _gen_reverse_type_match(t1, t2):
    """Reverse of the operation above.
    """
    if (
        isinstance(t1, types.Array)
        and isinstance(t2, types.Array)
        and isinstance(t1.dtype, types.Float)
        and isinstance(t2.dtype, types.Integer)
    ):
        return ".astype(np.{})".format(t1.dtype)
    return ""


def _get_table_parallel_flags(join_node, array_dists):
    par_dists = (
        distributed_pass.Distribution.OneD,
        distributed_pass.Distribution.OneD_Var,
    )

    left_parallel = all(
        array_dists[v.name] in par_dists for v in join_node.left_vars.values()
    )
    right_parallel = all(
        array_dists[v.name] in par_dists for v in join_node.right_vars.values()
    )
    if not left_parallel:
        assert not any(
            array_dists[v.name] in par_dists for v in join_node.left_vars.values()
        )
    if not right_parallel:
        assert not any(
            array_dists[v.name] in par_dists for v in join_node.right_vars.values()
        )

    if left_parallel or right_parallel:
        assert all(
            array_dists[v.name] in par_dists for v in join_node.df_out_vars.values()
        )

    return left_parallel, right_parallel

def _gen_local_hash_join(left_key_names, right_key_names, left_other_names, right_other_names, is_left, is_right):
    print("gen_local_hash_join: is_left=", is_left, " is_right=", is_right)
    n_keys = len(left_key_names)
    func_text =  "    # beginning of _gen_local_hash_join\n"
    eList = []
    for i in range(n_keys):
        eList.append("t1_keys[{}]".format(i))
    for i in range(n_keys):
        eList.append("t2_keys[{}]".format(i))
    for i in range(len(left_other_names)):
        eList.append("data_left[{}]".format(i))
    for i in range(len(right_other_names)):
        eList.append("data_right[{}]".format(i))
    func_text += "    info_list_total = [{}]\n".format(",".join("array_to_info({})".format(a) for a in eList))
    func_text += "    table_total = arr_info_list_to_table(info_list_total)\n"

    func_text += "    out_table = hash_join_table(table_total, {}, {}, {}, {}, {})\n".format(n_keys, len(left_other_names), len(right_other_names), is_left, is_right)
    func_text += "    delete_table(table_total)\n"
    use_cpp_code = True
    if use_cpp_code == False:
        func_text += (
            "    out_t1_keys, out_t2_keys, out_data_left, out_data_right"
            " = bodo.ir.join.local_hash_join(t1_keys, t2_keys, data_left, data_right, {}, {})\n".format(
                is_left, is_right
            )
        )
    if use_cpp_code:
        idx = 0
        for i, t in enumerate(left_other_names):
            func_text += "    left_{} = info_to_array(info_from_table(out_table, {}), {})\n".format(i,idx,t)
            idx += 1
        for i, t in enumerate(right_other_names):
            func_text += "    right_{} = info_to_array(info_from_table(out_table, {}), {})\n".format(i,idx,t)
            idx += 1
        for i, t in enumerate(left_key_names):
            func_text += "    t1_keys_{} = info_to_array(info_from_table(out_table, {}), {})\n".format(i,idx,t)
            idx += 1
        for i, t in enumerate(right_key_names):
            func_text += "    t2_keys_{} = info_to_array(info_from_table(out_table, {}), {})\n".format(i,idx,t)
            idx += 1
        func_text += "    delete_table(out_table)\n"
    return func_text

def _gen_par_shuffle(
    key_names, data_names, key_tup_out, data_tup_out, key_types, other_key_types
):
    """Generates data shuffle.
    Converts data to C arrays, creates a C table, shuffles it, and returns
    regular arrays.
    """
    key_names = tuple(
        "{}{}".format(key_names[i], _gen_type_match(key_types[i], other_key_types[i]))
        for i in range(len(key_names))
    )
    all_arrs = key_names + data_names
    n_keys = len(key_names)
    n_data = len(data_names)
    n_all = len(all_arrs)

    # convert arrays to table
    func_text = "    info_list = [{}]\n".format(
        ", ".join("array_to_info({})".format(a) for a in all_arrs)
    )
    func_text += "    table = arr_info_list_to_table(info_list)\n"

    # shuffle
    func_text += "    out_table = shuffle_table(table, {})\n".format(n_keys)

    # extract arrays from output table
    out_keys = [
        "info_to_array(info_from_table(out_table, {}), {})".format(i, all_arrs[i])
        for i in range(n_keys)
    ]
    out_data = [
        "info_to_array(info_from_table(out_table, {}), {})".format(i, all_arrs[i])
        for i in range(n_keys, n_all)
    ]
    func_text += "    {} = ({},)\n".format(key_tup_out, ", ".join(out_keys))
    func_text += "    {} = ({}{})\n".format(
        data_tup_out, ", ".join(out_data), "," if n_data != 0 else ""
    )
    # func_text += "    print(bodo.get_rank(), {})\n".format(key_tup_out)
    # func_text += "    print(bodo.get_rank(), {})\n".format(data_tup_out)

    # clean up
    func_text += "    delete_table(table)\n"
    func_text += "    delete_table(out_table)\n"

    return func_text


# @numba.njit
def parallel_join_impl(key_arrs, data):
    # alloc shuffle meta
    n_pes = bodo.libs.distributed_api.get_size()
    pre_shuffle_meta = alloc_pre_shuffle_metadata(key_arrs, data, n_pes, False)
    n = len(key_arrs[0])
    node_ids = np.empty(n, np.int32)

    # calc send/recv counts
    for i in range(n):
        val = getitem_arr_tup_single(key_arrs, i)
        node_id = hash(val) % n_pes
        node_ids[i] = node_id
        update_shuffle_meta(pre_shuffle_meta, node_id, i, key_arrs, data, False)

    shuffle_meta = finalize_shuffle_meta(key_arrs, data, pre_shuffle_meta, n_pes, False)

    # write send buffers
    for i in range(n):
        node_id = node_ids[i]
        write_send_buff(shuffle_meta, node_id, i, key_arrs, data)
        # update last since it is reused in data
        shuffle_meta.tmp_offset[node_id] += 1

    # shuffle
    recvs = alltoallv_tup(key_arrs + data, shuffle_meta, key_arrs)
    out_keys = _get_keys_tup(recvs, key_arrs)
    out_data = _get_data_tup(recvs, key_arrs)
    return out_keys, out_data


@generated_jit(nopython=True, cache=True)
def parallel_shuffle(key_arrs, data):
    return parallel_join_impl


@numba.njit
def parallel_asof_comm(left_key_arrs, right_key_arrs, right_data):
    # align the left and right intervals
    # allgather the boundaries of all left intervals and calculate overlap
    # rank = bodo.libs.distributed_api.get_rank()
    n_pes = bodo.libs.distributed_api.get_size()
    # TODO: multiple keys
    bnd_starts = np.empty(n_pes, left_key_arrs[0].dtype)
    bnd_ends = np.empty(n_pes, left_key_arrs[0].dtype)
    bodo.libs.distributed_api.allgather(bnd_starts, left_key_arrs[0][0])
    bodo.libs.distributed_api.allgather(bnd_ends, left_key_arrs[0][-1])

    send_counts = np.zeros(n_pes, np.int32)
    send_disp = np.zeros(n_pes, np.int32)
    recv_counts = np.zeros(n_pes, np.int32)
    my_start = right_key_arrs[0][0]
    my_end = right_key_arrs[0][-1]

    offset = -1
    i = 0
    # ignore no overlap processors (end of their interval is before current)
    while i < n_pes - 1 and bnd_ends[i] < my_start:
        i += 1
    while i < n_pes and bnd_starts[i] <= my_end:
        offset, count = _count_overlap(right_key_arrs[0], bnd_starts[i], bnd_ends[i])
        # one extra element in case first value is needed for start of boundary
        if offset != 0:
            offset -= 1
            count += 1
        send_counts[i] = count
        send_disp[i] = offset
        i += 1
    # one extra element in case last value is need for start of boundary
    # TODO: see if next processor provides the value
    while i < n_pes:
        send_counts[i] = 1
        send_disp[i] = len(right_key_arrs[0]) - 1
        i += 1

    bodo.libs.distributed_api.alltoall(send_counts, recv_counts, 1)
    n_total_recv = recv_counts.sum()
    out_r_keys = np.empty(n_total_recv, right_key_arrs[0].dtype)
    # TODO: support string
    out_r_data = alloc_arr_tup(n_total_recv, right_data)
    recv_disp = bodo.ir.join.calc_disp(recv_counts)
    bodo.libs.distributed_api.alltoallv(
        right_key_arrs[0], out_r_keys, send_counts, recv_counts, send_disp, recv_disp
    )
    bodo.libs.distributed_api.alltoallv_tup(
        right_data, out_r_data, send_counts, recv_counts, send_disp, recv_disp
    )

    return (out_r_keys,), out_r_data


@numba.njit
def _count_overlap(r_key_arr, start, end):
    # TODO: use binary search
    count = 0
    offset = 0
    j = 0
    while j < len(r_key_arr) and r_key_arr[j] < start:
        offset += 1
        j += 1
    while j < len(r_key_arr) and start <= r_key_arr[j] <= end:
        j += 1
        count += 1
    return offset, count


def write_send_buff(shuffle_meta, node_id, i, key_arrs, data):
    return i


@overload(write_send_buff)
def write_data_buff_overload(meta, node_id, i, key_arrs, data):
    func_text = "def f(meta, node_id, i, key_arrs, data):\n"
    func_text += "  w_ind = meta.send_disp[node_id] + meta.tmp_offset[node_id]\n"
    n_keys = len(key_arrs.types)
    for i, typ in enumerate(key_arrs.types + data.types):
        arr = "key_arrs[{}]".format(i) if i < n_keys else "data[{}]".format(i - n_keys)
        if not typ in (string_type, string_array_type):
            func_text += "  meta.send_buff_tup[{}][w_ind] = {}[i]\n".format(i, arr)
        else:
            func_text += "  n_chars_{} = get_str_arr_item_length({}, i)\n".format(
                i, arr
            )
            func_text += "  meta.send_arr_lens_tup[{}][w_ind] = n_chars_{}\n".format(
                i, i
            )
            if i >= n_keys:
                func_text += "  out_bitmap = meta.send_arr_nulls_tup[{}][meta.send_disp_nulls[node_id]:].ctypes\n".format(
                    i
                )
                func_text += "  bit_val = get_bit_bitmap(get_null_bitmap_ptr(data[{}]), i)\n".format(
                    i - n_keys
                )
                func_text += (
                    "  set_bit_to(out_bitmap, meta.tmp_offset[node_id], bit_val)\n"
                )
            func_text += "  indc_{} = meta.send_disp_char_tup[{}][node_id] + meta.tmp_offset_char_tup[{}][node_id]\n".format(
                i, i, i
            )
            func_text += "  item_ptr_{} = get_str_arr_item_ptr({}, i)\n".format(i, arr)
            func_text += "  str_copy_ptr(meta.send_arr_chars_tup[{}], indc_{}, item_ptr_{}, n_chars_{})\n".format(
                i, i, i, i
            )
            func_text += "  meta.tmp_offset_char_tup[{}][node_id] += n_chars_{}\n".format(
                i, i
            )

    func_text += "  return w_ind\n"

    # print(func_text)

    loc_vars = {}
    exec(
        func_text,
        {
            "str_copy_ptr": str_copy_ptr,
            "get_null_bitmap_ptr": get_null_bitmap_ptr,
            "get_bit_bitmap": get_bit_bitmap,
            "set_bit_to": set_bit_to,
            "get_str_arr_item_length": get_str_arr_item_length,
            "get_str_arr_item_ptr": get_str_arr_item_ptr,
        },
        loc_vars,
    )
    write_impl = loc_vars["f"]
    return write_impl


from numba.typing.templates import signature, AbstractTemplate, infer_global, infer
from numba.extending import register_model, models, lower_builtin
from numba import cgutils


from llvmlite import ir as lir
import llvmlite.binding as ll
from numba.targets.arrayobj import make_array
from bodo.utils.utils import _numba_to_c_type_map
from bodo.libs import hdist

ll.add_symbol("c_alltoallv", hdist.c_alltoallv)


@numba.njit
def calc_disp(arr):
    disp = np.empty_like(arr)
    disp[0] = 0
    for i in range(1, len(arr)):
        disp[i] = disp[i - 1] + arr[i - 1]
    return disp


def ensure_capacity(arr, new_size):
    new_arr = arr
    curr_len = len(arr)
    if curr_len < new_size:
        new_len = 2 * curr_len
        new_arr = bodo.hiframes.pd_categorical_ext.fix_cat_array_type(
            bodo.utils.utils.alloc_type(new_len, arr)
        )
        new_arr[:curr_len] = arr
    return new_arr


@overload(ensure_capacity)
def ensure_capacity_overload(arr, new_size):
    if isinstance(arr, types.Array) or arr == boolean_array:
        return ensure_capacity
    assert isinstance(arr, (types.Tuple, types.UniTuple))
    count = arr.count

    func_text = "def f(arr, new_size):\n"
    func_text += "  return ({}{})\n".format(
        ",".join(
            ["ensure_capacity(arr[{}], new_size)".format(i) for i in range(count)]
        ),
        "," if count == 1 else "",
    )  # single value needs comma to become tuple

    loc_vars = {}
    exec(func_text, {"ensure_capacity": ensure_capacity}, loc_vars)
    alloc_impl = loc_vars["f"]
    return alloc_impl


@numba.njit
def ensure_capacity_str(arr, new_size, n_chars):
    # new_size is right after write index
    new_arr = arr
    curr_len = len(arr)
    curr_num_chars = num_total_chars(arr)
    needed_total_chars = getitem_str_offset(arr, new_size - 1) + n_chars

    # TODO: corner case test
    # print("new alloc", new_size, curr_len, getitem_str_offset(arr, new_size-1), n_chars, curr_num_chars)
    if curr_len < new_size or needed_total_chars > curr_num_chars:
        new_len = int(2 * curr_len if curr_len < new_size else curr_len)
        new_num_chars = int(
            2 * curr_num_chars + n_chars
            if needed_total_chars > curr_num_chars
            else curr_num_chars
        )
        new_arr = pre_alloc_string_array(new_len, new_num_chars)
        copy_str_arr_slice(new_arr, arr, new_size - 1)

    return new_arr


def trim_arr_tup(data, new_size):  # pragma: no cover
    return data


@overload(trim_arr_tup)
def trim_arr_tup_overload(data, new_size):
    assert isinstance(data, (types.Tuple, types.UniTuple))
    count = data.count

    func_text = "def f(data, new_size):\n"
    func_text += "  return ({}{})\n".format(
        ",".join(["trim_arr(data[{}], new_size)".format(i) for i in range(count)]),
        "," if count == 1 else "",
    )  # single value needs comma to become tuple

    loc_vars = {}
    exec(func_text, {"trim_arr": trim_arr}, loc_vars)
    alloc_impl = loc_vars["f"]
    return alloc_impl


# @numba.njit
# def copy_merge_data(left_key, data_left, data_right, left_ind, right_ind,
#                         out_left_key, out_data_left, out_data_right, out_ind):
#     out_left_key = ensure_capacity(out_left_key, out_ind+1)
#     out_data_left = ensure_capacity(out_data_left, out_ind+1)
#     out_data_right = ensure_capacity(out_data_right, out_ind+1)

#     out_left_key[out_ind] = left_keys[left_ind]
#     copyElement_tup(data_left, left_ind, out_data_left, out_ind)
#     copyElement_tup(data_right, right_ind, out_data_right, out_ind)
#     return out_left_key, out_data_left, out_data_right


def copy_elem_buff(arr, ind, val):  # pragma: no cover
    new_arr = ensure_capacity(arr, ind + 1)
    new_arr[ind] = val
    return new_arr


@overload(copy_elem_buff)
def copy_elem_buff_overload(arr, ind, val):
    if isinstance(arr, types.Array) or arr == boolean_array:
        return copy_elem_buff

    assert arr == string_array_type

    def copy_elem_buff_str(arr, ind, val):
        new_arr = ensure_capacity_str(arr, ind + 1, get_utf8_size(val))
        new_arr[ind] = val
        return new_arr

    return copy_elem_buff_str


def copy_elem_buff_tup(arr, ind, val):  # pragma: no cover
    return arr


@overload(copy_elem_buff_tup)
def copy_elem_buff_tup_overload(data, ind, val):
    assert isinstance(data, (types.Tuple, types.UniTuple))
    count = data.count

    func_text = "def f(data, ind, val):\n"
    for i in range(count):
        func_text += "  arr_{} = copy_elem_buff(data[{}], ind, val[{}])\n".format(
            i, i, i
        )
    func_text += "  return ({}{})\n".format(
        ",".join(["arr_{}".format(i) for i in range(count)]), "," if count == 1 else ""
    )

    loc_vars = {}
    exec(func_text, {"copy_elem_buff": copy_elem_buff}, loc_vars)
    cp_impl = loc_vars["f"]
    return cp_impl


def trim_arr(arr, size):  # pragma: no cover
    return bodo.hiframes.pd_categorical_ext.fix_cat_array_type(arr[:size])


@overload(trim_arr)
def trim_arr_overload(arr, size):
    if isinstance(arr, types.Array) or arr == boolean_array:
        return trim_arr

    assert arr == string_array_type

    def trim_arr_str(arr, size):
        # print("trim size", size, arr[size-1], getitem_str_offset(arr, size))
        new_arr = pre_alloc_string_array(size, np.int64(getitem_str_offset(arr, size)))
        copy_str_arr_slice(new_arr, arr, size)
        return new_arr

    return trim_arr_str


def setnan_elem_buff(arr, ind):  # pragma: no cover
    new_arr = ensure_capacity(arr, ind + 1)
    setitem_arr_nan(new_arr, ind)
    return new_arr


@overload(setnan_elem_buff)
def setnan_elem_buff_overload(arr, ind):
    if isinstance(arr, types.Array) or arr == boolean_array:
        return setnan_elem_buff

    assert arr == string_array_type

    def setnan_elem_buff_str(arr, ind):
        new_arr = ensure_capacity_str(arr, ind + 1, 0)
        # TODO: why doesn't setitem_str_offset work
        # setitem_str_offset(arr, ind+1, getitem_str_offset(arr, ind))
        new_arr[ind] = ""
        setitem_arr_nan(new_arr, ind)
        # print(getitem_str_offset(arr, ind), getitem_str_offset(arr, ind+1))
        return new_arr

    return setnan_elem_buff_str


def setnan_elem_buff_tup(arr, ind):  # pragma: no cover
    return arr


@overload(setnan_elem_buff_tup)
def setnan_elem_buff_tup_overload(data, ind):
    assert isinstance(data, (types.Tuple, types.UniTuple))
    count = data.count

    func_text = "def f(data, ind):\n"
    for i in range(count):
        func_text += "  arr_{} = setnan_elem_buff(data[{}], ind)\n".format(i, i)
    func_text += "  return ({}{})\n".format(
        ",".join(["arr_{}".format(i) for i in range(count)]), "," if count == 1 else ""
    )

    loc_vars = {}
    exec(func_text, {"setnan_elem_buff": setnan_elem_buff}, loc_vars)
    cp_impl = loc_vars["f"]
    return cp_impl


# @numba.njit
def local_hash_join_impl(
    left_keys, right_keys, data_left, data_right, is_left=False, is_right=False
):
    l_len = len(left_keys[0])
    r_len = len(right_keys[0])
    print("l_len=", l_len, " r_len=", r_len, " is_left=", is_left, " is_right=", is_right)
    # TODO: approximate output size properly
    curr_size = 101 + min(l_len, r_len) // 2
    if is_left:
        curr_size = int(1.1 * l_len)
    if is_right:
        curr_size = int(1.1 * r_len)
    if is_left and is_right:
        curr_size = int(1.1 * (l_len + r_len))

    out_left_key = alloc_arr_tup(curr_size, left_keys)
    out_data_left = alloc_arr_tup(curr_size, data_left)
    out_data_right = alloc_arr_tup(curr_size, data_right)
    # keep track of matched keys in case of right join
    if is_right:
        r_matched = np.full(r_len, False, np.bool_)

    out_ind = 0
    m = bodo.libs.dict_ext.multimap_int64_init()
    for i in range(r_len):
        # store hash if keys are tuple or non-int
        k = _hash_if_tup(getitem_arr_tup(right_keys, i))
        bodo.libs.dict_ext.multimap_int64_insert(m, k, i)

    r = bodo.libs.dict_ext.multimap_int64_equal_range_alloc()
    for i in range(l_len):
        l_key = getitem_arr_tup(left_keys, i)
        l_data_val = getitem_arr_tup(data_left, i)
        l_data_nan_bits = get_nan_bits_tup(data_left, i)
        k = _hash_if_tup(l_key)
        bodo.libs.dict_ext.multimap_int64_equal_range_inplace(m, k, r)
        num_matched = 0
        for j in r:
            # if hash for stored, check left key against the actual right key
            r_ind = _check_ind_if_hashed(right_keys, j, l_key)
            if r_ind == -1:
                continue
            if is_right:
                r_matched[r_ind] = True
            out_left_key = copy_elem_buff_tup(out_left_key, out_ind, l_key)
            r_data_val = getitem_arr_tup(data_right, r_ind)
            r_data_nan_bits = get_nan_bits_tup(data_right, r_ind)
            out_data_right = copy_elem_buff_tup(out_data_right, out_ind, r_data_val)
            set_nan_bits_tup(out_data_right, out_ind, r_data_nan_bits)
            out_data_left = copy_elem_buff_tup(out_data_left, out_ind, l_data_val)
            set_nan_bits_tup(out_data_left, out_ind, l_data_nan_bits)
            out_ind += 1
            num_matched += 1
        if is_left and num_matched == 0:
            out_left_key = copy_elem_buff_tup(out_left_key, out_ind, l_key)
            out_data_left = copy_elem_buff_tup(out_data_left, out_ind, l_data_val)
            set_nan_bits_tup(out_data_left, out_ind, l_data_nan_bits)
            out_data_right = setnan_elem_buff_tup(out_data_right, out_ind)
            out_ind += 1

    bodo.libs.dict_ext.multimap_int64_equal_range_dealloc(r)

    # produce NA rows for unmatched right keys
    if is_right:
        for i in range(r_len):
            if not r_matched[i]:
                r_key = getitem_arr_tup(right_keys, i)
                r_data_val = getitem_arr_tup(data_right, i)
                r_data_nan_bits = get_nan_bits_tup(data_right, i)
                out_left_key = copy_elem_buff_tup(out_left_key, out_ind, r_key)
                out_data_right = copy_elem_buff_tup(out_data_right, out_ind, r_data_val)
                set_nan_bits_tup(out_data_right, out_ind, r_data_nan_bits)
                out_data_left = setnan_elem_buff_tup(out_data_left, out_ind)
                out_ind += 1

    out_left_key = trim_arr_tup(out_left_key, out_ind)

    out_right_key = copy_arr_tup(out_left_key)
    out_data_left = trim_arr_tup(out_data_left, out_ind)
    out_data_right = trim_arr_tup(out_data_right, out_ind)

    return out_left_key, out_right_key, out_data_left, out_data_right


@generated_jit(nopython=True, cache=True, no_cpython_wrapper=True)
def local_hash_join(
    left_keys, right_keys, data_left, data_right, is_left=False, is_right=False
):
    return local_hash_join_impl


@generated_jit(nopython=True, cache=True)
def _hash_if_tup(val):
    if val == types.Tuple((types.intp,)):
        return lambda val: val[0]
    return lambda val: hash(val)


@generated_jit(nopython=True, cache=True)
def _check_ind_if_hashed(right_keys, r_ind, l_key):
    if right_keys == types.Tuple((types.intp[::1],)):
        return lambda right_keys, r_ind, l_key: r_ind

    def _impl(right_keys, r_ind, l_key):
        r_key = getitem_arr_tup(right_keys, r_ind)
        if r_key != l_key:
            return -1
        return r_ind

    return _impl


def _gen_pd_join(
    left_key_vars,
    right_key_vars,
    left_other_col_vars,
    right_other_col_vars,
    typemap,
    how,
):
    l_keys_typ = types.Tuple([typemap[v.name] for v in left_key_vars])
    r_keys_typ = types.Tuple([typemap[v.name] for v in right_key_vars])
    l_data_typ = types.Tuple([typemap[v.name] for v in left_other_col_vars])
    r_data_typ = types.Tuple([typemap[v.name] for v in right_other_col_vars])
    lk_name = "pd_join{}".format(str(ir_utils.next_label()))
    rk_name = "pd_join{}".format(str(ir_utils.next_label()))
    ld_name = "pd_join{}".format(str(ir_utils.next_label()))
    rd_name = "pd_join{}".format(str(ir_utils.next_label()))
    setattr(types, lk_name, l_keys_typ)
    setattr(types, rk_name, r_keys_typ)
    setattr(types, ld_name, l_data_typ)
    setattr(types, rd_name, r_data_typ)

    typ_strs = "out_t1_keys='{}', out_t2_keys='{}', out_data_left='{}', out_data_right='{}'".format(
        lk_name, rk_name, ld_name, rd_name
    )

    func_text = "def f(t1_keys, t2_keys, data_left, data_right, how, same_keys):\n"
    func_text += "  with objmode({}):\n".format(typ_strs)
    func_text += "    out_t1_keys, out_t2_keys, out_data_left, out_data_right = bodo.ir.join.pd_join(t1_keys, t2_keys, data_left, data_right, how, same_keys)\n"
    func_text += "  return out_t1_keys, out_t2_keys, out_data_left, out_data_right\n"

    loc_vars = {}
    exec(func_text, {"bodo": bodo, "objmode": objmode}, loc_vars)
    f = loc_vars["f"]

    # TODO: no_cpython_wrapper=True crashes for some reason
    jit_func = numba.njit(f)
    bodo.ir.csv_ext.compiled_funcs.append(jit_func)
    return jit_func


def pd_join(t1_keys, t2_keys, data_left, data_right, how, same_keys):
    # construct dataframes and call join
    lk_prefix = "lk"
    rk_prefix = "rk"
    if same_keys:
        # same key case matters since Pandas avoids NAs in outer case
        # like test_join_outer_seq1
        rk_prefix = "lk"

    data1 = {"{}{}".format(lk_prefix, i): v for i, v in enumerate(t1_keys)}
    data2 = {"{}{}".format(rk_prefix, i): v for i, v in enumerate(t2_keys)}
    left_on = list(data1.keys())
    right_on = list(data2.keys())
    data1.update({"ld{}".format(i): v for i, v in enumerate(data_left)})
    data2.update({"rd{}".format(i): v for i, v in enumerate(data_right)})
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = df1.merge(df2, left_on=left_on, right_on=right_on, how=how)
    out_t1_keys = tuple(
        _get_pd_out_arr(df3["{}{}".format(lk_prefix, i)], t1_keys[i])
        for i in range(len(t1_keys))
    )
    out_t2_keys = tuple(
        _get_pd_out_arr(df3["{}{}".format(rk_prefix, i)], t2_keys[i])
        for i in range(len(t2_keys))
    )
    out_data_left = tuple(
        _get_pd_out_arr(df3["ld{}".format(i)], data_left[i])
        for i in range(len(data_left))
    )
    out_data_right = tuple(
        _get_pd_out_arr(df3["rd{}".format(i)], data_right[i])
        for i in range(len(data_right))
    )
    return out_t1_keys, out_t2_keys, out_data_left, out_data_right


def _get_pd_out_arr(out_series, in_arr):
    # Pandas converts int to float in case of NAs so convert back
    if out_series.hasnans and in_arr.dtype in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ):
        out_series = out_series.fillna(0)
    return out_series.astype(in_arr.dtype).values


@numba.njit
def local_merge_new(
    left_keys, right_keys, data_left, data_right, is_left=False, is_outer=False
):
    l_len = len(left_keys[0])
    r_len = len(right_keys[0])
    # TODO: approximate output size properly
    curr_size = 101 + min(l_len, r_len) // 2
    if is_left:
        curr_size = int(1.1 * l_len)
    if is_outer:
        curr_size = int(1.1 * r_len)
    if is_left and is_outer:
        curr_size = int(1.1 * (l_len + r_len))

    out_left_key = alloc_arr_tup(curr_size, left_keys)
    out_data_left = alloc_arr_tup(curr_size, data_left)
    out_data_right = alloc_arr_tup(curr_size, data_right)

    out_ind = 0
    left_ind = 0
    right_ind = 0

    while left_ind < len(left_keys[0]) and right_ind < len(right_keys[0]):
        if getitem_arr_tup(left_keys, left_ind) == getitem_arr_tup(
            right_keys, right_ind
        ):
            key = getitem_arr_tup(left_keys, left_ind)
            # catesian product in case of duplicate keys on either side
            left_run = left_ind
            while (
                left_run < len(left_keys[0])
                and getitem_arr_tup(left_keys, left_run) == key
            ):
                right_run = right_ind
                while (
                    right_run < len(right_keys[0])
                    and getitem_arr_tup(right_keys, right_run) == key
                ):
                    out_left_key = copy_elem_buff_tup(out_left_key, out_ind, key)
                    l_data_val = getitem_arr_tup(data_left, left_run)
                    out_data_left = copy_elem_buff_tup(
                        out_data_left, out_ind, l_data_val
                    )
                    r_data_val = getitem_arr_tup(data_right, right_run)
                    out_data_right = copy_elem_buff_tup(
                        out_data_right, out_ind, r_data_val
                    )
                    out_ind += 1
                    right_run += 1
                left_run += 1
            left_ind = left_run
            right_ind = right_run
        elif getitem_arr_tup(left_keys, left_ind) < getitem_arr_tup(
            right_keys, right_ind
        ):
            if is_left:
                out_left_key = copy_elem_buff_tup(
                    out_left_key, out_ind, getitem_arr_tup(left_keys, left_ind)
                )
                l_data_val = getitem_arr_tup(data_left, left_ind)
                out_data_left = copy_elem_buff_tup(out_data_left, out_ind, l_data_val)
                out_data_right = setnan_elem_buff_tup(out_data_right, out_ind)
                out_ind += 1
            left_ind += 1
        else:
            if is_outer:
                # TODO: support separate keys?
                out_left_key = copy_elem_buff_tup(
                    out_left_key, out_ind, getitem_arr_tup(right_keys, right_ind)
                )
                out_data_left = setnan_elem_buff_tup(out_data_left, out_ind)
                r_data_val = getitem_arr_tup(data_right, right_ind)
                out_data_right = copy_elem_buff_tup(out_data_right, out_ind, r_data_val)
                out_ind += 1
            right_ind += 1

    if is_left and left_ind < len(left_keys[0]):
        while left_ind < len(left_keys[0]):
            out_left_key = copy_elem_buff_tup(
                out_left_key, out_ind, getitem_arr_tup(left_keys, left_ind)
            )
            l_data_val = getitem_arr_tup(data_left, left_ind)
            out_data_left = copy_elem_buff_tup(out_data_left, out_ind, l_data_val)
            out_data_right = setnan_elem_buff_tup(out_data_right, out_ind)
            out_ind += 1
            left_ind += 1

    if is_outer and right_ind < len(right_keys[0]):
        while right_ind < len(right_keys[0]):
            out_left_key = copy_elem_buff_tup(
                out_left_key, out_ind, getitem_arr_tup(right_keys, right_ind)
            )
            out_data_left = setnan_elem_buff_tup(out_data_left, out_ind)
            r_data_val = getitem_arr_tup(data_right, right_ind)
            out_data_right = copy_elem_buff_tup(out_data_right, out_ind, r_data_val)
            out_ind += 1
            right_ind += 1

    # out_left_key = out_left_key[:out_ind]
    out_left_key = trim_arr_tup(out_left_key, out_ind)

    out_right_key = copy_arr_tup(out_left_key)
    out_data_left = trim_arr_tup(out_data_left, out_ind)
    out_data_right = trim_arr_tup(out_data_right, out_ind)

    return out_left_key, out_right_key, out_data_left, out_data_right


@numba.njit
def local_merge_asof(left_keys, right_keys, data_left, data_right):
    # adapted from pandas/_libs/join_func_helper.pxi
    l_size = len(left_keys[0])
    r_size = len(right_keys[0])

    out_left_keys = alloc_arr_tup(l_size, left_keys)
    out_right_keys = alloc_arr_tup(l_size, right_keys)
    out_data_left = alloc_arr_tup(l_size, data_left)
    out_data_right = alloc_arr_tup(l_size, data_right)

    left_ind = 0
    right_ind = 0

    for left_ind in range(l_size):
        # restart right_ind if it went negative in a previous iteration
        if right_ind < 0:
            right_ind = 0

        # find last position in right whose value is less than left's
        while right_ind < r_size and getitem_arr_tup(
            right_keys, right_ind
        ) <= getitem_arr_tup(left_keys, left_ind):
            right_ind += 1

        right_ind -= 1

        setitem_arr_tup(out_left_keys, left_ind, getitem_arr_tup(left_keys, left_ind))
        # TODO: copy_tup
        setitem_arr_tup(out_data_left, left_ind, getitem_arr_tup(data_left, left_ind))

        if right_ind >= 0:
            setitem_arr_tup(
                out_right_keys, left_ind, getitem_arr_tup(right_keys, right_ind)
            )
            setitem_arr_tup(
                out_data_right, left_ind, getitem_arr_tup(data_right, right_ind)
            )
        else:
            setitem_arr_tup_nan(out_right_keys, left_ind)
            setitem_arr_tup_nan(out_data_right, left_ind)

    return out_left_keys, out_right_keys, out_data_left, out_data_right


def setitem_arr_nan(arr, ind, int_nan_const=0):
    arr[ind] = np.nan


@overload(setitem_arr_nan)
def setitem_arr_nan_overload(arr, ind, int_nan_const=0):
    if isinstance(arr.dtype, types.Float):
        return setitem_arr_nan

    if isinstance(arr.dtype, (types.NPDatetime, types.NPTimedelta)):
        nat = arr.dtype("NaT")

        def _setnan_impl(arr, ind, int_nan_const=0):
            arr[ind] = nat

        return _setnan_impl

    if arr == string_array_type:
        return lambda arr, ind, int_nan_const=0: str_arr_set_na(arr, ind)

    if isinstance(arr, IntegerArrayType) or arr == boolean_array:
        return lambda arr, ind, int_nan_const=0: bodo.libs.int_arr_ext.set_bit_to_arr(
            arr._null_bitmap, ind, 0
        )

    # TODO: support strings, bools, etc.
    # XXX: set NA values in bool arrays to False
    # FIXME: replace with proper NaN
    if arr.dtype == types.bool_:

        def b_set(arr, ind, int_nan_const=0):
            arr[ind] = False

        return b_set

    if isinstance(arr, CategoricalArray):

        def setitem_arr_nan_cat(arr, ind, int_nan_const=0):
            int_arr = bodo.hiframes.pd_categorical_ext.cat_array_to_int(arr)
            int_arr[ind] = -1

        return setitem_arr_nan_cat

    # XXX set integer NA to 0 to avoid unexpected errors
    # TODO: convert integer to float if nan
    if isinstance(arr.dtype, types.Integer):

        def setitem_arr_nan_int(arr, ind, int_nan_const=0):
            arr[ind] = int_nan_const

        return setitem_arr_nan_int

    return lambda arr, ind, int_nan_const: None


def setitem_arr_tup_nan(arr_tup, ind, int_nan_const=0):  # pragma: no cover
    for arr in arr_tup:
        arr[ind] = np.nan


@overload(setitem_arr_tup_nan)
def setitem_arr_tup_nan_overload(arr_tup, ind, int_nan_const=0):
    count = arr_tup.count

    func_text = "def f(arr_tup, ind, int_nan_const=0):\n"
    for i in range(count):
        func_text += "  setitem_arr_nan(arr_tup[{}], ind, int_nan_const)\n".format(i)
    func_text += "  return\n"

    loc_vars = {}
    exec(func_text, {"setitem_arr_nan": setitem_arr_nan}, loc_vars)
    impl = loc_vars["f"]
    return impl


def copy_arr_tup(arrs):
    return tuple(a.copy() for a in arrs)


@overload(copy_arr_tup)
def copy_arr_tup_overload(arrs):
    count = arrs.count
    func_text = "def f(arrs):\n"
    func_text += "  return ({},)\n".format(
        ",".join("arrs[{}].copy()".format(i) for i in range(count))
    )

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["f"]
    return impl


def get_nan_bits(arr, ind):  # pragma: no cover
    return 0


@overload(get_nan_bits)
def overload_get_nan_bits(arr, ind):
    """Get nan bit for types that have null bitmap
    """
    if arr == string_array_type:

        def impl_str(arr, ind):
            in_null_bitmap_ptr = get_null_bitmap_ptr(arr)
            return get_bit_bitmap(in_null_bitmap_ptr, ind)

        return impl_str

    if isinstance(arr, IntegerArrayType) or arr == boolean_array:

        def impl(arr, ind):
            return bodo.libs.int_arr_ext.get_bit_bitmap_arr(arr._null_bitmap, ind)

        return impl

    return lambda arr, ind: False


def get_nan_bits_tup(arr_tup, ind):  # pragma: no cover
    return tuple(get_nan_bits(arr, ind) for arr in arr_tup)


@overload(get_nan_bits_tup)
def overload_get_nan_bits_tup(arr_tup, ind):
    count = arr_tup.count

    func_text = "def f(arr_tup, ind):\n"
    func_text += "  return ({}{})\n".format(
        ",".join(["get_nan_bits(arr_tup[{}], ind)".format(i) for i in range(count)]),
        "," if count == 1 else "",
    )  # single value needs comma to become tuple

    loc_vars = {}
    exec(func_text, {"get_nan_bits": get_nan_bits}, loc_vars)
    impl = loc_vars["f"]
    return impl


def set_nan_bits(arr, ind, na_val):  # pragma: no cover
    return 0


@overload(set_nan_bits)
def overload_set_nan_bits(arr, ind, na_val):
    """Set nan bit for types that have null bitmap, currently just string array
    """
    if arr == string_array_type:

        def impl_str(arr, ind, na_val):
            in_null_bitmap_ptr = get_null_bitmap_ptr(arr)
            set_bit_to(in_null_bitmap_ptr, ind, na_val)

        return impl_str

    if isinstance(arr, IntegerArrayType) or arr == boolean_array:

        def impl(arr, ind, na_val):
            bodo.libs.int_arr_ext.set_bit_to_arr(arr._null_bitmap, ind, na_val)

        return impl
    return lambda arr, ind, na_val: None


def set_nan_bits_tup(arr_tup, ind, na_val):  # pragma: no cover
    return tuple(set_nan_bits(arr, ind, na_val) for arr in arr_tup)


@overload(set_nan_bits_tup)
def overload_set_nan_bits_tup(arr_tup, ind, na_val):
    count = arr_tup.count

    func_text = "def f(arr_tup, ind, na_val):\n"
    for i in range(count):
        func_text += "  set_nan_bits(arr_tup[{}], ind, na_val[{}])\n".format(i, i)
    func_text += "  return\n"

    loc_vars = {}
    exec(func_text, {"set_nan_bits": set_nan_bits}, loc_vars)
    impl = loc_vars["f"]
    return impl
