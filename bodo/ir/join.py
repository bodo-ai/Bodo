# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""IR node for the join and merge"""
from collections import defaultdict

import numba
import numpy as np
import pandas as pd
from numba import generated_jit
from numba.core import ir, ir_utils, typeinfer, types
from numba.core.extending import intrinsic
from numba.core.ir_utils import (
    compile_to_numba_ir,
    replace_arg_nodes,
    replace_vars_inner,
    visit_vars_inner,
)
from numba.extending import overload

import bodo
from bodo.libs.array import (
    arr_info_list_to_table,
    array_to_info,
    compute_node_partition_by_hash,
    delete_table,
    delete_table_decref_arrays,
    hash_join_table,
    info_from_table,
    info_to_array,
    shuffle_table,
)
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.int_arr_ext import IntDtype, IntegerArrayType
from bodo.libs.str_arr_ext import (
    copy_str_arr_slice,
    cp_str_list_to_array,
    get_bit_bitmap,
    get_null_bitmap_ptr,
    get_str_arr_item_length,
    get_str_arr_item_ptr,
    get_utf8_size,
    getitem_str_offset,
    num_total_chars,
    pre_alloc_string_array,
    set_bit_to,
    str_copy_ptr,
    string_array_type,
    to_list_if_immutable_arr,
)
from bodo.libs.str_ext import string_type
from bodo.libs.timsort import getitem_arr_tup, setitem_arr_tup
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.transforms.distributed_analysis import Distribution
from bodo.utils.shuffle import (
    _get_data_tup,
    _get_keys_tup,
    alloc_pre_shuffle_metadata,
    alltoallv_tup,
    finalize_shuffle_meta,
    getitem_arr_tup_single,
    update_shuffle_meta,
)
from bodo.utils.typing import (
    BodoError,
    dtype_to_array_type,
    find_common_np_dtype,
    is_dtype_nullable,
    is_nullable_type,
    to_nullable_type,
)
from bodo.utils.utils import alloc_arr_tup, debug_prints


class Join(ir.Stmt):
    def __init__(
        self,
        df_out,
        left_df,
        right_df,
        left_keys,
        right_keys,
        out_data_vars,
        left_vars,
        right_vars,
        how,
        suffix_x,
        suffix_y,
        loc,
        is_left,
        is_right,
        is_join,
        left_index,
        right_index,
        indicator,
        is_na_equal,
        gen_cond_expr,
    ):
        self.df_out = df_out
        self.left_df = left_df
        self.right_df = right_df
        self.left_keys = left_keys
        self.right_keys = right_keys
        self.out_data_vars = out_data_vars
        self.left_vars = left_vars
        self.right_vars = right_vars
        self.how = how
        self.suffix_x = suffix_x
        self.suffix_y = suffix_y
        self.loc = loc
        self.is_left = is_left
        self.is_right = is_right
        self.is_join = is_join
        self.left_index = left_index
        self.right_index = right_index
        self.indicator = indicator
        self.is_na_equal = is_na_equal
        self.gen_cond_expr = gen_cond_expr
        # find columns used in general join condition to avoid removing them in rm dead
        self.left_cond_cols = set(
            c for c in left_vars.keys() if f"(left.{c})" in gen_cond_expr
        )
        self.right_cond_cols = set(
            c for c in right_vars.keys() if f"(right.{c})" in gen_cond_expr
        )
        # keep the origin of output columns to enable proper dead code elimination
        # columns with common name that are not common keys will get a suffix
        comm_keys = set(left_keys) & set(right_keys)
        comm_data = set(left_vars.keys()) & set(right_vars.keys())
        add_suffix = comm_data - comm_keys
        # vect_same_key is a vector of boolean containing whether the key have the same
        # name on the left and right. This has impact how they show up in the output:
        # ---If they have the same name then they show up just once (and have no additional
        #   missing entry)
        # ---If they have different name then they show up two times (and can have additional
        #   missing entry)
        vect_same_key = []
        n_keys = len(left_keys)
        for iKey in range(n_keys):
            name_left = left_keys[iKey]
            name_right = right_keys[iKey]
            vect_same_key.append(name_left == name_right)
        self.vect_same_key = vect_same_key
        #
        self.column_origins = {
            (str(c) + suffix_x if c in add_suffix else c): ("left", c)
            for c in left_vars.keys()
        }
        self.column_origins.update(
            {
                (str(c) + suffix_y if c in add_suffix else c): ("right", c)
                for c in right_vars.keys()
            }
        )

    def __repr__(self):  # pragma: no cover
        out_cols = ""
        for (c, v) in self.out_data_vars.items():
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
    assert len(join_node.out_data_vars) > 0, "empty join in array analysis"

    # arrays of left_df and right_df have same size in first dimension
    all_shapes = []
    in_vars = list(join_node.left_vars.values())
    for col_var in in_vars:
        typ = typemap[col_var.name]
        col_shape = equiv_set.get_shape(col_var)
        if col_shape:
            all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    all_shapes = []
    in_vars = list(join_node.right_vars.values())
    for col_var in in_vars:
        typ = typemap[col_var.name]
        col_shape = equiv_set.get_shape(col_var)
        if col_shape:
            all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variable for an output column
    all_shapes = []
    for col_var in join_node.out_data_vars.values():
        typ = typemap[col_var.name]
        shape = array_analysis._gen_shape_call(equiv_set, col_var, typ.ndim, None, post)
        equiv_set.insert_equiv(col_var, shape)
        all_shapes.append(shape[0])
        equiv_set.define(col_var, set())

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post


numba.parfors.array_analysis.array_analysis_extensions[Join] = join_array_analysis


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
    for col_var in join_node.out_data_vars.values():
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
    for col_var in join_node.out_data_vars.values():
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
    comm_keys = set(join_node.left_keys) & set(join_node.right_keys)
    comm_data = set(join_node.left_vars.keys()) & set(join_node.right_vars.keys())
    add_suffix = comm_data - comm_keys
    for out_col_name, out_col_var in join_node.out_data_vars.items():
        # left suffix
        if join_node.indicator and out_col_name == "_merge":
            continue
        if not out_col_name in join_node.column_origins:
            raise BodoError(
                "join(): The variable " + out_col_name + " is absent from the output"
            )
        ePair = join_node.column_origins[out_col_name]
        if ePair[0] == "left":
            col_var = join_node.left_vars[ePair[1]]
        else:
            col_var = join_node.right_vars[ePair[1]]
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
    for col_name in list(join_node.out_data_vars.keys()):
        join_node.out_data_vars[col_name] = visit_vars_inner(
            join_node.out_data_vars[col_name], callback, cbdata
        )


# add call to visit Join variable
ir_utils.visit_vars_extensions[Join] = visit_vars_join


def remove_dead_join(
    join_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):
    # if an output column is dead, the related input column is not needed
    # anymore in the join
    dead_cols = []
    # TODO: remove output of dead keys
    all_cols_dead = True

    for col_name, col_var in join_node.out_data_vars.items():
        if col_var.name in lives:
            all_cols_dead = False
            continue
        # avoid index sentinel (that is not in column_origins)
        if col_name == "$_bodo_index_":
            continue
        # avoid indicator (that is not in column_origins)
        if join_node.indicator and col_name == "_merge":
            dead_cols.append("_merge")
            # If _merge is removed, switch indicator to False so we don't expect
            # to generate indicator code.
            join_node.indicator = False
            continue
        orig, orig_name = join_node.column_origins[col_name]
        if (
            orig == "left"
            and orig_name not in join_node.left_keys
            and orig_name not in join_node.left_cond_cols
        ):
            join_node.left_vars.pop(orig_name)
            dead_cols.append(col_name)
        if (
            orig == "right"
            and orig_name not in join_node.right_keys
            and orig_name not in join_node.right_cond_cols
        ):
            join_node.right_vars.pop(orig_name)
            dead_cols.append(col_name)

    for cname in dead_cols:
        join_node.out_data_vars.pop(cname)

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
    def_set.update({v.name for v in join_node.out_data_vars.values()})

    return numba.core.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.core.analysis.ir_extension_usedefs[Join] = join_usedefs


def get_copies_join(join_node, typemap):
    # join doesn't generate copies, it just kills the output columns
    kill_set = set(v.name for v in join_node.out_data_vars.values())
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
    for col_name in list(join_node.out_data_vars.keys()):
        join_node.out_data_vars[col_name] = replace_vars_inner(
            join_node.out_data_vars[col_name], var_dict
        )

    return


ir_utils.apply_copy_propagate_extensions[Join] = apply_copies_join


def build_join_definitions(join_node, definitions=None):
    if definitions is None:
        definitions = defaultdict(list)

    for col_var in join_node.out_data_vars.values():
        definitions[col_var.name].append(join_node)

    return definitions


ir_utils.build_defs_extensions[Join] = build_join_definitions


def join_distributed_run(
    join_node, array_dists, typemap, calltypes, typingctx, targetctx
):
    left_parallel, right_parallel = False, False
    if array_dists is not None:
        left_parallel, right_parallel = _get_table_parallel_flags(
            join_node, array_dists
        )

    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    n_keys = len(join_node.left_keys)
    # get column variables
    left_key_vars = tuple(join_node.left_vars[c] for c in join_node.left_keys)
    right_key_vars = tuple(join_node.right_vars[c] for c in join_node.right_keys)

    left_columns = tuple(join_node.left_vars.keys())
    right_columns = tuple(join_node.right_vars.keys())
    # Optional column refer: When doing a merge on column and index, the key
    # is put also in output, so we need one additional column in that case.
    optional_col_var = ()
    optional_key_tuple = ()
    optional_column = False
    if join_node.left_index and not join_node.right_index and not join_node.is_join:
        optional_key = join_node.right_keys[0]
        if optional_key in left_columns:
            optional_key_tuple = (optional_key,)
            optional_col_var = (join_node.right_vars[optional_key],)
            optional_column = True

    if join_node.right_index and not join_node.left_index and not join_node.is_join:
        optional_key = join_node.left_keys[0]
        if optional_key in right_columns:
            optional_key_tuple = (optional_key,)
            optional_col_var = (join_node.left_vars[optional_key],)
            optional_column = True

    # It is a fairly complex construction
    # ---keys can have same name on left and right.
    # ---keys can be two times or one time in output.
    # ---output keys can be computed from just one column (and so additional NaN may occur)
    #  or from two columns
    # ---keys may be from the index or not.
    #
    # The following rules apply:
    # ---A key that is an index or not behave in the same way. If it is an index key
    #  then its name is just the fancy "$_bodo_index_", so please don't use that one.
    # ---Identity of key is determined by their name, whether they are index or not.
    # ---If a key appears on same name on left and right then both columns are used
    #  and so the name will never have additional NaNs

    out_optional_key_vars = tuple(
        join_node.out_data_vars[cname] for cname in optional_key_tuple
    )
    left_other_col_vars = tuple(
        v
        for (n, v) in sorted(join_node.left_vars.items(), key=lambda a: str(a[0]))
        if n not in join_node.left_keys
    )
    right_other_col_vars = tuple(
        v
        for (n, v) in sorted(join_node.right_vars.items(), key=lambda a: str(a[0]))
        if n not in join_node.right_keys
    )
    # get column types
    arg_vars = (
        optional_col_var
        + left_key_vars
        + right_key_vars
        + left_other_col_vars
        + right_other_col_vars
    )
    arg_typs = tuple(typemap[v.name] for v in arg_vars)

    # arg names of non-key columns
    optional_names = tuple("opti_c" + str(i) for i in range(len(optional_col_var)))

    left_other_names = tuple("t1_c" + str(i) for i in range(len(left_other_col_vars)))
    right_other_names = tuple("t2_c" + str(i) for i in range(len(right_other_col_vars)))
    left_other_types = tuple([typemap[c.name] for c in left_other_col_vars])
    right_other_types = tuple([typemap[c.name] for c in right_other_col_vars])
    left_key_names = tuple("t1_key" + str(i) for i in range(n_keys))
    right_key_names = tuple("t2_key" + str(i) for i in range(n_keys))
    glbs = {}
    loc = join_node.loc

    func_text = "def f({}{}, {},{}{}{}):\n".format(
        ("{},".format(optional_names[0]) if len(optional_names) == 1 else ""),
        ",".join(left_key_names),
        ",".join(right_key_names),
        ",".join(left_other_names),
        ("," if len(left_other_names) != 0 else ""),
        ",".join(right_other_names),
    )

    left_key_types = tuple(typemap[v.name] for v in left_key_vars)
    right_key_types = tuple(typemap[v.name] for v in right_key_vars)

    # add common key type to globals to use below for type conversion
    for i in range(n_keys):
        glbs[f"key_type_{i}"] = _match_join_key_types(
            left_key_types[i], right_key_types[i], loc
        )

    func_text += "    t1_keys = ({},)\n".format(
        ", ".join(
            f"bodo.utils.utils.astype({left_key_names[i]}, key_type_{i})"
            for i in range(n_keys)
        )
    )
    func_text += "    t2_keys = ({},)\n".format(
        ", ".join(
            f"bodo.utils.utils.astype({right_key_names[i]}, key_type_{i})"
            for i in range(n_keys)
        )
    )

    func_text += "    data_left = ({}{})\n".format(
        ",".join(left_other_names), "," if len(left_other_names) != 0 else ""
    )
    func_text += "    data_right = ({}{})\n".format(
        ",".join(right_other_names), "," if len(right_other_names) != 0 else ""
    )
    out_keys = []
    for cname in join_node.left_keys:
        if str(cname) + join_node.suffix_x in join_node.out_data_vars:
            cname_work = str(cname) + join_node.suffix_x
        else:
            cname_work = cname
        out_keys.append(join_node.out_data_vars[cname_work])
    for i, cname in enumerate(join_node.right_keys):
        if not join_node.vect_same_key[i] and not join_node.is_join:
            cname_work = str(cname) + join_node.suffix_y
            if not cname_work in join_node.out_data_vars:
                cname_work = cname
            assert cname_work in join_node.out_data_vars
            out_keys.append(join_node.out_data_vars[cname_work])

    def _get_out_col_var(cname, is_left):
        if is_left and str(cname) + join_node.suffix_x in join_node.out_data_vars:
            return join_node.out_data_vars[str(cname) + join_node.suffix_x]
        if not is_left and str(cname) + join_node.suffix_y in join_node.out_data_vars:
            return join_node.out_data_vars[str(cname) + join_node.suffix_y]

        return join_node.out_data_vars[cname]

    merge_out = out_optional_key_vars + tuple(out_keys)
    merge_out += tuple(
        _get_out_col_var(n, True)
        for (n, v) in sorted(join_node.left_vars.items(), key=lambda a: str(a[0]))
        if n not in join_node.left_keys
    )
    merge_out += tuple(
        _get_out_col_var(n, False)
        for (n, v) in sorted(join_node.right_vars.items(), key=lambda a: str(a[0]))
        if n not in join_node.right_keys
    )
    if join_node.indicator:
        merge_out += (_get_out_col_var("_merge", False),)
    out_names = ["t3_c" + str(i) for i in range(len(merge_out))]

    # Generate a general join condition function if it exists
    # and determine the data columns it needs.
    cfunc_cond_address, left_col_nums, right_col_nums = _gen_general_cond_cfunc(
        join_node, typemap
    )

    if join_node.how == "asof":
        if left_parallel or right_parallel:
            assert left_parallel and right_parallel
            # only the right key needs to be aligned
            func_text += "    t2_keys, data_right = parallel_asof_comm(t1_keys, t2_keys, data_right)\n"
        func_text += (
            "    out_t1_keys, out_t2_keys, out_data_left, out_data_right"
            " = bodo.ir.join.local_merge_asof(t1_keys, t2_keys, data_left, data_right)\n"
        )
    else:
        func_text += _gen_local_hash_join(
            optional_column,
            left_key_names,
            right_key_names,
            left_key_types,
            right_key_types,
            left_other_names,
            right_other_names,
            left_other_types,
            right_other_types,
            join_node.vect_same_key,
            join_node.is_left,
            join_node.is_right,
            join_node.is_join,
            left_parallel,
            right_parallel,
            glbs,
            [typemap[v.name] for v in merge_out],
            join_node.loc,
            join_node.indicator,
            join_node.is_na_equal,
            left_col_nums,
            right_col_nums,
        )
    if join_node.how == "asof":
        for i in range(len(left_other_names)):
            func_text += "    left_{} = out_data_left[{}]\n".format(i, i)
        for i in range(len(right_other_names)):
            func_text += "    right_{} = out_data_right[{}]\n".format(i, i)
        for i in range(n_keys):
            func_text += f"    t1_keys_{i} = out_t1_keys[{i}]\n"
        for i in range(n_keys):
            func_text += f"    t2_keys_{i} = out_t2_keys[{i}]\n"

    idx = 0
    if optional_column:
        func_text += f"    {out_names[idx]} = opti_0\n"
        idx += 1
    for i in range(n_keys):
        func_text += f"    {out_names[idx]} = t1_keys_{i}\n"
        idx += 1
    for i in range(n_keys):
        if not join_node.vect_same_key[i] and not join_node.is_join:
            func_text += f"    {out_names[idx]} = t2_keys_{i}\n"
            idx += 1
    for i in range(len(left_other_names)):
        func_text += f"    {out_names[idx]} = left_{i}\n"
        idx += 1
    for i in range(len(right_other_names)):
        func_text += f"    {out_names[idx]} = right_{i}\n"
        idx += 1
    if join_node.indicator:
        func_text += f"    {out_names[idx]} = indicator_col\n"
        idx += 1
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    join_impl = loc_vars["f"]

    glbs.update(
        {
            "bodo": bodo,
            "np": np,
            "pd": pd,
            "to_list_if_immutable_arr": to_list_if_immutable_arr,
            "cp_str_list_to_array": cp_str_list_to_array,
            "parallel_asof_comm": parallel_asof_comm,
            "array_to_info": array_to_info,
            "arr_info_list_to_table": arr_info_list_to_table,
            "shuffle_table": shuffle_table,
            "hash_join_table": hash_join_table,
            "info_from_table": info_from_table,
            "info_to_array": info_to_array,
            "delete_table": delete_table,
            "delete_table_decref_arrays": delete_table_decref_arrays,
            "cfunc_cond": cfunc_cond_address,
        }
    )
    f_block = compile_to_numba_ir(
        join_impl,
        glbs,
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=arg_typs,
        typemap=typemap,
        calltypes=calltypes,
    ).blocks.popitem()[1]
    replace_arg_nodes(f_block, arg_vars)

    nodes = f_block.body[:-3]
    for i in range(len(merge_out)):
        nodes[-len(merge_out) + i].target = merge_out[i]
    return nodes


distributed_pass.distributed_run_extensions[Join] = join_distributed_run


def _gen_general_cond_cfunc(join_node, typemap):
    """Generate cfunc for general join condition and return its address.
    Return 0 (NULL) if there is no general join condition to evaluate.
    The cfunc takes data pointers of table columns and row indices to access as input
    and returns True or False.
    E.g. left_table=[A_data_ptr, B_data_ptr], right_table=[A_data_ptr, C_data_ptr],
    left_ind=3, right_ind=7
    """
    expr = join_node.gen_cond_expr
    if not expr:
        return 0, [], []

    # get column name to table column index
    left_col_to_ind = _get_col_to_ind(join_node.left_keys, join_node.left_vars)
    right_col_to_ind = _get_col_to_ind(join_node.right_keys, join_node.right_vars)

    table_getitem_funcs = {}
    func_text = "def f(left_table, right_table, left_ind, right_ind):\n"
    expr, func_text, left_col_nums = _replace_column_accesses(
        expr,
        left_col_to_ind,
        typemap,
        join_node.left_vars,
        table_getitem_funcs,
        func_text,
        "left",
        len(join_node.left_keys),
    )
    expr, func_text, right_col_nums = _replace_column_accesses(
        expr,
        right_col_to_ind,
        typemap,
        join_node.right_vars,
        table_getitem_funcs,
        func_text,
        "right",
        len(join_node.right_keys),
    )
    func_text += f"  return {expr}"

    loc_vars = {}
    exec(func_text, table_getitem_funcs, loc_vars)
    cond_func = loc_vars["f"]

    c_sig = types.bool_(types.voidptr, types.voidptr, types.int64, types.int64)
    cfunc_cond = numba.cfunc(c_sig, nopython=True)(cond_func)
    return cfunc_cond.address, left_col_nums, right_col_nums


def _replace_column_accesses(
    expr,
    col_to_ind,
    typemap,
    col_vars,
    table_getitem_funcs,
    func_text,
    table_name,
    n_keys,
):
    """replace column accesses in join condition expression with an intrinsic that loads
    values from table data pointers.
    For example, left.B is replaced with data_ptrs[1][row_ind]

    This function returns the modified expression, the func_text defining the column
    accesses, and the list of column numbers that are used by the table.
    """
    col_nums = []
    for c, c_ind in col_to_ind.items():
        cname = f"({table_name}.{c})"
        if cname not in expr:
            continue
        getitem_fname = f"getitem_{table_name}_val_{c_ind}"
        val_varname = f"_bodo_{table_name}_val_{c_ind}"
        func_text += (
            f"  {val_varname} = {getitem_fname}({table_name}_table, {table_name}_ind)\n"
        )
        col_dtype = typemap[col_vars[c].name].dtype
        table_getitem_funcs[getitem_fname] = _gen_row_access_intrinsic(col_dtype, c_ind)
        expr = expr.replace(cname, val_varname)
        # only append the column if it is not a key
        if c_ind >= n_keys:
            col_nums.append(c_ind)
    return expr, func_text, col_nums


def _get_col_to_ind(key_names, col_vars):
    """create a mapping from input dataframe column names to column indices in the C++
    left/right table structure (keys are first, then data columns).
    """
    n_keys = len(key_names)
    col_to_ind = {c: i for (i, c) in enumerate(key_names)}
    i = n_keys
    for c in sorted(col_vars, key=lambda a: str(a)):
        if c in key_names:
            continue
        col_to_ind[c] = i
        i += 1
    return col_to_ind


def _gen_row_access_intrinsic(col_dtype, c_ind):
    """Generate an intrinsic for loading a value from a table column with 'col_dtype'
    data type. 'c_ind' is the index of the column within the table.
    The intrinsic's input is an array of data pointers for the table's data, and a row
    index.

    For example, col_dtype=int64, c_ind=1, table=[A_data_ptr, B_data_ptr, C_data_ptr],
    row_ind=2 will return 6 for the table below.
    A  B  C
    1  4  7
    2  5  8
    3  6  9
    """
    from llvmlite import ir as lir

    @intrinsic
    def getitem_func(typingctx, table_t, ind_t):
        def codegen(context, builder, signature, args):
            table, row_ind = args
            # cast void* to void**
            table = builder.bitcast(table, lir.IntType(8).as_pointer().as_pointer())
            # get data pointer for input column and cast to proper data type
            col_ind = lir.Constant(lir.IntType(64), c_ind)
            col_ptr = builder.load(builder.gep(table, [col_ind]))
            col_ptr = builder.bitcast(
                col_ptr, context.get_data_type(col_dtype).as_pointer()
            )
            return builder.load(builder.gep(col_ptr, [row_ind]))

        return col_dtype(types.voidptr, types.int64), codegen

    return getitem_func


def _match_join_key_types(t1, t2, loc):
    """make sure join key array types match since required in the C++ join code"""
    if t1 == t2:
        return t1

    try:
        arr = dtype_to_array_type(find_common_np_dtype([t1, t2]))
        # output should be nullable if any input is nullable
        return (
            to_nullable_type(arr)
            if is_nullable_type(t1) or is_nullable_type(t2)
            else arr
        )
    except:
        raise BodoError(f"Join key types {t1} and {t2} do not match", loc=loc)


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
            array_dists[v.name] in par_dists for v in join_node.out_data_vars.values()
        )

    return left_parallel, right_parallel


def _gen_local_hash_join(
    optional_column,
    left_key_names,
    right_key_names,
    left_key_types,
    right_key_types,
    left_other_names,
    right_other_names,
    left_other_types,
    right_other_types,
    vect_same_key,
    is_left,
    is_right,
    is_join,
    left_parallel,
    right_parallel,
    glbs,
    out_types,
    loc,
    indicator,
    is_na_equal,
    left_col_nums,
    right_col_nums,
):
    # In some case the column in output has a type different from the one in input.
    # TODO: Unify those type changes between all cases.
    def needs_typechange(in_type, need_nullable, is_same_key):
        return (
            isinstance(in_type, types.Array)
            and not is_dtype_nullable(in_type.dtype)
            and need_nullable
            and not is_same_key
        )

    # The vect_need_typechange is computed in the python code and is sent to C++.
    # This is a general approach for this kind of combinatorial problem: compute in python
    # preferably to C++. Compute in dataframe_pass.py preferably to the IR node.
    #
    # The vect_need_typechange is for the need to change the type in some cases.
    # Following constraints have to be taken into account:
    # ---For NullableArrayType the output column has the same format as the input column
    # ---For numpy array of float the output column has the same format as the input column
    # ---For numpy array of integer it may happen that we need to add missing entries and so
    #  we change the output type.
    # ---For categorical array data, the input is integer and we do not change the type.
    #   We may have to change this if missing data in categorical gets treated differently.

    vect_need_typechange = []
    for i in range(len(left_key_names)):
        key_type = _match_join_key_types(left_key_types[i], right_key_types[i], loc)
        vect_need_typechange.append(
            needs_typechange(key_type, is_right, vect_same_key[i])
        )
    for i in range(len(left_other_names)):
        vect_need_typechange.append(
            needs_typechange(left_other_types[i], is_right, False)
        )
    for i in range(len(right_key_names)):
        if not vect_same_key[i] and not is_join:
            key_type = _match_join_key_types(left_key_types[i], right_key_types[i], loc)
            vect_need_typechange.append(needs_typechange(key_type, is_left, False))
    for i in range(len(right_other_names)):
        vect_need_typechange.append(
            needs_typechange(right_other_types[i], is_left, False)
        )

    def get_out_type(idx, in_type, in_name, need_nullable, is_same_key):
        if (
            isinstance(in_type, types.Array)
            and not is_dtype_nullable(in_type.dtype)
            and need_nullable
            and not is_same_key
        ):
            # the only types with another nullable version are int and bool currently
            if isinstance(in_type.dtype, types.Integer):
                int_typ_name = IntDtype(in_type.dtype).name
                assert int_typ_name.endswith("Dtype()")
                int_typ_name = int_typ_name[:-7]
                ins_text = f'    typ_{idx} = bodo.hiframes.pd_series_ext.get_series_data(pd.Series([1], dtype="{int_typ_name}"))\n'
                out_type = f"typ_{idx}"
            else:
                assert (
                    in_type.dtype == types.bool_
                ), "unexpected non-nullable type in join"
                ins_text = (
                    f"    typ_{idx} = bodo.libs.bool_arr_ext.alloc_bool_array(1)\n"
                )
                out_type = f"typ_{idx}"
        else:
            ins_text = ""
            out_type = in_name
        return (ins_text, out_type)

    n_keys = len(left_key_names)
    func_text = "    # beginning of _gen_local_hash_join\n"
    eList_l = []
    for i in range(n_keys):
        eList_l.append("t1_keys[{}]".format(i))
    for i in range(len(left_other_names)):
        eList_l.append("data_left[{}]".format(i))
    func_text += "    info_list_total_l = [{}]\n".format(
        ",".join("array_to_info({})".format(a) for a in eList_l)
    )
    func_text += "    table_left = arr_info_list_to_table(info_list_total_l)\n"
    eList_r = []
    for i in range(n_keys):
        eList_r.append("t2_keys[{}]".format(i))
    for i in range(len(right_other_names)):
        eList_r.append("data_right[{}]".format(i))
    func_text += "    info_list_total_r = [{}]\n".format(
        ",".join("array_to_info({})".format(a) for a in eList_r)
    )
    func_text += "    table_right = arr_info_list_to_table(info_list_total_r)\n"
    func_text += "    vect_same_key = np.array([{}])\n".format(
        ",".join("1" if x else "0" for x in vect_same_key)
    )
    func_text += "    vect_need_typechange = np.array([{}])\n".format(
        ",".join("1" if x else "0" for x in vect_need_typechange)
    )
    func_text += f"    left_table_cond_columns = np.array({left_col_nums if len(left_col_nums) > 0 else [-1]}, dtype=np.int64)\n"
    func_text += f"    right_table_cond_columns = np.array({right_col_nums if len(right_col_nums) > 0 else [-1]}, dtype=np.int64)\n"
    func_text += "    out_table = hash_join_table(table_left, table_right, {}, {}, {}, {}, {}, vect_same_key.ctypes, vect_need_typechange.ctypes, {}, {}, {}, {}, {}, {}, cfunc_cond, left_table_cond_columns.ctypes, {}, right_table_cond_columns.ctypes, {})\n".format(
        left_parallel,
        right_parallel,
        n_keys,
        len(left_other_names),
        len(right_other_names),
        is_left,
        is_right,
        is_join,
        optional_column,
        indicator,
        is_na_equal,
        len(left_col_nums),
        len(right_col_nums),
    )
    func_text += "    delete_table(table_left)\n"
    func_text += "    delete_table(table_right)\n"
    idx = 0
    if optional_column:
        func_text += (
            f"    opti_0 = info_to_array(info_from_table(out_table, {idx}), opti_c0)\n"
        )
        idx += 1
    for i, t in enumerate(left_key_names):
        key_type = _match_join_key_types(left_key_types[i], right_key_types[i], loc)
        rec_typ = get_out_type(
            idx, key_type, f"t1_keys[{i}]", is_right, vect_same_key[i]
        )
        func_text += rec_typ[0]
        glbs[f"out_type_{idx}"] = out_types[idx]
        # use astype only if necessary due to Index handling bugs
        # see: test_merge_index_column_second"[df22-df10]" TODO(ehsan): fix
        if key_type != left_key_types[i]:
            func_text += f"    t1_keys_{i} = bodo.utils.utils.astype(info_to_array(info_from_table(out_table, {idx}), {rec_typ[1]}), out_type_{idx})\n"
        else:
            func_text += f"    t1_keys_{i} = info_to_array(info_from_table(out_table, {idx}), {rec_typ[1]})\n"
        idx += 1
    for i, t in enumerate(left_other_names):
        rec_typ = get_out_type(idx, left_other_types[i], t, is_right, False)
        func_text += rec_typ[0]
        func_text += (
            "    left_{} = info_to_array(info_from_table(out_table, {}), {})\n".format(
                i, idx, rec_typ[1]
            )
        )
        idx += 1
    for i, t in enumerate(right_key_names):
        if not vect_same_key[i] and not is_join:
            key_type = _match_join_key_types(left_key_types[i], right_key_types[i], loc)
            rec_typ = get_out_type(idx, key_type, f"t2_keys[{i}]", is_left, False)
            func_text += rec_typ[0]
            # NOTE: subtracting len(left_other_names) since output right keys are
            # generated before left_other_names
            glbs[f"out_type_{idx}"] = out_types[idx - len(left_other_names)]
            if key_type != right_key_types[i]:
                func_text += f"    t2_keys_{i} = bodo.utils.utils.astype(info_to_array(info_from_table(out_table, {idx}), {rec_typ[1]}), out_type_{idx})\n"
            else:
                func_text += f"    t2_keys_{i} = info_to_array(info_from_table(out_table, {idx}), {rec_typ[1]})\n"
            idx += 1
    for i, t in enumerate(right_other_names):
        rec_typ = get_out_type(idx, right_other_types[i], t, is_left, False)
        func_text += rec_typ[0]
        func_text += (
            "    right_{} = info_to_array(info_from_table(out_table, {}), {})\n".format(
                i, idx, rec_typ[1]
            )
        )
        idx += 1
    if indicator:
        func_text += f"    typ_{idx} = pd.Categorical(values=['both'], categories=('left_only', 'right_only', 'both'))\n"
        func_text += f"    indicator_col = info_to_array(info_from_table(out_table, {idx}), typ_{idx})\n"
        idx += 1

    func_text += "    delete_table(out_table)\n"
    return func_text


# @numba.njit
def parallel_join_impl(key_arrs, data):  # pragma: no cover
    # alloc shuffle meta
    n_pes = bodo.libs.distributed_api.get_size()
    pre_shuffle_meta = alloc_pre_shuffle_metadata(key_arrs, data, n_pes, False)
    n = len(key_arrs[0])
    node_ids = np.empty(n, np.int32)

    input_table = arr_info_list_to_table([array_to_info(key_arrs[0])])
    n_key = 1
    out_table = compute_node_partition_by_hash(input_table, n_key, n_pes)
    data_dummy = np.empty(1, np.int32)
    out_arr = info_to_array(info_from_table(out_table, 0), data_dummy)
    delete_table(out_table)
    delete_table(input_table)

    # calc send/recv counts
    for i in range(n):
        val = getitem_arr_tup_single(key_arrs, i)
        node_id = out_arr[i]
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
def parallel_asof_comm(left_key_arrs, right_key_arrs, right_data):  # pragma: no cover
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
def _count_overlap(r_key_arr, start, end):  # pragma: no cover
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


def write_send_buff(shuffle_meta, node_id, i, key_arrs, data):  # pragma: no cover
    return i


@overload(write_send_buff, no_unliteral=True)
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
            func_text += (
                "  meta.tmp_offset_char_tup[{}][node_id] += n_chars_{}\n".format(i, i)
            )

    func_text += "  return w_ind\n"

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


import llvmlite.binding as ll

from bodo.libs import hdist

ll.add_symbol("c_alltoallv", hdist.c_alltoallv)


@numba.njit
def calc_disp(arr):  # pragma: no cover
    disp = np.empty_like(arr)
    disp[0] = 0
    for i in range(1, len(arr)):
        disp[i] = disp[i - 1] + arr[i - 1]
    return disp


def ensure_capacity(arr, new_size):  # pragma: no cover
    new_arr = arr
    curr_len = len(arr)
    if curr_len < new_size:
        new_len = 2 * curr_len
        new_arr = bodo.utils.utils.alloc_type(new_len, arr)
        new_arr[:curr_len] = arr
    return new_arr


@overload(ensure_capacity, no_unliteral=True)
def ensure_capacity_overload(arr, new_size):
    if isinstance(arr, types.Array) or arr == boolean_array:
        return ensure_capacity
    assert isinstance(arr, types.BaseTuple)
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
def ensure_capacity_str(arr, new_size, n_chars):  # pragma: no cover
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


@overload(trim_arr_tup, no_unliteral=True)
def trim_arr_tup_overload(data, new_size):
    assert isinstance(data, types.BaseTuple)
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


@overload(copy_elem_buff, no_unliteral=True)
def copy_elem_buff_overload(arr, ind, val):
    if isinstance(arr, types.Array) or arr == boolean_array:
        return copy_elem_buff

    assert arr == string_array_type

    def copy_elem_buff_str(arr, ind, val):  # pragma: no cover
        new_arr = ensure_capacity_str(arr, ind + 1, get_utf8_size(val))
        new_arr[ind] = val
        return new_arr

    return copy_elem_buff_str


def copy_elem_buff_tup(arr, ind, val):  # pragma: no cover
    return arr


@overload(copy_elem_buff_tup, no_unliteral=True)
def copy_elem_buff_tup_overload(data, ind, val):
    assert isinstance(data, types.BaseTuple)
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
    return arr[:size]


@overload(trim_arr, no_unliteral=True)
def trim_arr_overload(arr, size):
    if isinstance(arr, types.Array) or arr == boolean_array:
        return trim_arr

    assert arr == string_array_type

    def trim_arr_str(arr, size):  # pragma: no cover
        # print("trim size", size, arr[size-1], getitem_str_offset(arr, size))
        new_arr = pre_alloc_string_array(size, np.int64(getitem_str_offset(arr, size)))
        copy_str_arr_slice(new_arr, arr, size)
        return new_arr

    return trim_arr_str


def setnan_elem_buff(arr, ind):  # pragma: no cover
    new_arr = ensure_capacity(arr, ind + 1)
    bodo.libs.array_kernels.setna(new_arr, ind)
    return new_arr


@overload(setnan_elem_buff, no_unliteral=True)
def setnan_elem_buff_overload(arr, ind):
    if isinstance(arr, types.Array) or arr == boolean_array:
        return setnan_elem_buff

    assert arr == string_array_type

    def setnan_elem_buff_str(arr, ind):  # pragma: no cover
        new_arr = ensure_capacity_str(arr, ind + 1, 0)
        # TODO: why doesn't setitem_str_offset work
        # setitem_str_offset(arr, ind+1, getitem_str_offset(arr, ind))
        new_arr[ind] = ""
        bodo.libs.array_kernels.setna(new_arr, ind)
        # print(getitem_str_offset(arr, ind), getitem_str_offset(arr, ind+1))
        return new_arr

    return setnan_elem_buff_str


def setnan_elem_buff_tup(arr, ind):  # pragma: no cover
    return arr


@overload(setnan_elem_buff_tup, no_unliteral=True)
def setnan_elem_buff_tup_overload(data, ind):
    assert isinstance(data, types.BaseTuple)
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


@numba.njit
def local_merge_asof(left_keys, right_keys, data_left, data_right):  # pragma: no cover
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
            bodo.libs.array_kernels.setna_tup(out_right_keys, left_ind)
            bodo.libs.array_kernels.setna_tup(out_data_right, left_ind)

    return out_left_keys, out_right_keys, out_data_left, out_data_right


def copy_arr_tup(arrs):  # pragma: no cover
    return tuple(a.copy() for a in arrs)


@overload(copy_arr_tup, no_unliteral=True)
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


@overload(get_nan_bits, no_unliteral=True)
def overload_get_nan_bits(arr, ind):
    """Get nan bit for types that have null bitmap"""
    if arr == string_array_type:

        def impl_str(arr, ind):  # pragma: no cover
            in_null_bitmap_ptr = get_null_bitmap_ptr(arr)
            return get_bit_bitmap(in_null_bitmap_ptr, ind)

        return impl_str

    if isinstance(arr, IntegerArrayType) or arr == boolean_array:

        def impl(arr, ind):  # pragma: no cover
            return bodo.libs.int_arr_ext.get_bit_bitmap_arr(arr._null_bitmap, ind)

        return impl

    return lambda arr, ind: False


def get_nan_bits_tup(arr_tup, ind):  # pragma: no cover
    return tuple(get_nan_bits(arr, ind) for arr in arr_tup)


@overload(get_nan_bits_tup, no_unliteral=True)
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


@overload(set_nan_bits, no_unliteral=True)
def overload_set_nan_bits(arr, ind, na_val):
    """Set nan bit for types that have null bitmap, currently just string array"""
    if arr == string_array_type:

        def impl_str(arr, ind, na_val):  # pragma: no cover
            in_null_bitmap_ptr = get_null_bitmap_ptr(arr)
            set_bit_to(in_null_bitmap_ptr, ind, na_val)

        return impl_str

    if isinstance(arr, IntegerArrayType) or arr == boolean_array:

        def impl(arr, ind, na_val):  # pragma: no cover
            bodo.libs.int_arr_ext.set_bit_to_arr(arr._null_bitmap, ind, na_val)

        return impl
    return lambda arr, ind, na_val: None


def set_nan_bits_tup(arr_tup, ind, na_val):  # pragma: no cover
    return tuple(set_nan_bits(arr, ind, na_val) for arr in arr_tup)


@overload(set_nan_bits_tup, no_unliteral=True)
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
