# Copyright (C) 2019 Bodo Inc. All rights reserved.
from collections import defaultdict
import numba
from numba import typeinfer, ir, ir_utils, config, types, cgutils
from numba.typing.templates import signature
from numba.extending import overload, intrinsic, register_model, models, box
from numba.ir_utils import (
    visit_vars_inner,
    replace_vars_inner,
    compile_to_numba_ir,
    replace_arg_nodes,
)
import bodo
from bodo.transforms import distributed_pass, distributed_analysis
from bodo.utils.utils import debug_prints, alloc_arr_tup
from bodo.transforms.distributed_analysis import Distribution
from bodo.libs.str_ext import string_type
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.timsort import copyElement_tup, getitem_arr_tup
from bodo import objmode
import pandas as pd
import numpy as np

from bodo.hiframes.pd_categorical_ext import PDCategoricalDtype, CategoricalArray


class ParquetReader(ir.Stmt):
    def __init__(
        self, file_name, df_out, col_names, col_indices, out_types, out_vars, loc
    ):
        self.file_name = file_name
        self.df_out = df_out  # used only for printing
        self.col_names = col_names
        self.col_indices = col_indices
        self.out_types = out_types
        self.out_vars = out_vars
        self.loc = loc

    def __repr__(self):  # pragma: no cover
        # TODO
        return "({}) = ReadParquet({}, {}, {}, {}, {})".format(
            self.df_out,
            self.file_name.name,
            self.col_names,
            self.col_indices,
            self.out_types,
            self.out_vars,
        )


def pq_array_analysis(pq_node, equiv_set, typemap, array_analysis):
    post = []
    # empty pq nodes should be deleted in remove dead
    assert len(pq_node.out_vars) > 0, "empty pq in array analysis"

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variable for an output column
    all_shapes = []
    for col_var in pq_node.out_vars:
        typ = typemap[col_var.name]
        (shape, c_post) = array_analysis._gen_shape_call(
            equiv_set, col_var, typ.ndim, None
        )
        equiv_set.insert_equiv(col_var, shape)
        post.extend(c_post)
        all_shapes.append(shape[0])
        equiv_set.define(col_var, {})

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post


numba.array_analysis.array_analysis_extensions[ParquetReader] = pq_array_analysis


def pq_distributed_analysis(pq_node, array_dists):
    for v in pq_node.out_vars:
        if v.name not in array_dists:
            array_dists[v.name] = Distribution.OneD

    return


distributed_analysis.distributed_analysis_extensions[
    ParquetReader
] = pq_distributed_analysis


def pq_typeinfer(pq_node, typeinferer):
    for col_var, typ in zip(pq_node.out_vars, pq_node.out_types):
        typeinferer.lock_type(col_var.name, typ, loc=pq_node.loc)
    return


typeinfer.typeinfer_extensions[ParquetReader] = pq_typeinfer


def visit_vars_pq(pq_node, callback, cbdata):
    if debug_prints():  # pragma: no cover
        print("visiting pq vars for:", pq_node)
        print("cbdata: ", sorted(cbdata.items()))

    # update output_vars
    new_out_vars = []
    for col_var in pq_node.out_vars:
        new_var = visit_vars_inner(col_var, callback, cbdata)
        new_out_vars.append(new_var)

    pq_node.out_vars = new_out_vars
    pq_node.file_name = visit_vars_inner(pq_node.file_name, callback, cbdata)
    return


# add call to visit pq variable
ir_utils.visit_vars_extensions[ParquetReader] = visit_vars_pq


def remove_dead_pq(pq_node, lives, arg_aliases, alias_map, func_ir, typemap):
    # TODO
    new_col_names = []
    new_out_vars = []
    new_out_types = []
    new_col_indices = []

    for i, col_var in enumerate(pq_node.out_vars):
        if col_var.name in lives:
            new_col_names.append(pq_node.col_names[i])
            new_out_vars.append(pq_node.out_vars[i])
            new_out_types.append(pq_node.out_types[i])
            new_col_indices.append(pq_node.col_indices[i])

    pq_node.col_names = new_col_names
    pq_node.out_vars = new_out_vars
    pq_node.out_types = new_out_types
    pq_node.col_indices = new_col_indices

    if len(pq_node.out_vars) == 0:
        return None

    return pq_node


ir_utils.remove_dead_extensions[ParquetReader] = remove_dead_pq


def pq_usedefs(pq_node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # output columns are defined
    def_set.update({v.name for v in pq_node.out_vars})
    use_set.add(pq_node.file_name.name)

    return numba.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.analysis.ir_extension_usedefs[ParquetReader] = pq_usedefs


def get_copies_pq(pq_node, typemap):
    # pq doesn't generate copies, it just kills the output columns
    kill_set = set(v.name for v in pq_node.out_vars)
    return set(), kill_set


ir_utils.copy_propagate_extensions[ParquetReader] = get_copies_pq


def apply_copies_pq(pq_node, var_dict, name_var_table, typemap, calltypes, save_copies):
    """apply copy propagate in pq node"""

    # update output_vars
    new_out_vars = []
    for col_var in pq_node.out_vars:
        new_var = replace_vars_inner(col_var, var_dict)
        new_out_vars.append(new_var)

    pq_node.out_vars = new_out_vars
    pq_node.file_name = replace_vars_inner(pq_node.file_name, var_dict)
    return


ir_utils.apply_copy_propagate_extensions[ParquetReader] = apply_copies_pq


def build_pq_definitions(pq_node, definitions=None):
    if definitions is None:
        definitions = defaultdict(list)

    for col_var in pq_node.out_vars:
        definitions[col_var.name].append(pq_node)

    return definitions


ir_utils.build_defs_extensions[ParquetReader] = build_pq_definitions
