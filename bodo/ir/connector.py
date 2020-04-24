# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Common IR extension functions for connectors such as CSV, Parquet and JSON readers.
"""
from collections import defaultdict
import numba
from numba.core import ir, ir_utils, typeinfer, types
from numba.extending import box, models, register_model
from numba.core.ir_utils import (
    visit_vars_inner,
    replace_vars_inner,
    compile_to_numba_ir,
    replace_arg_nodes,
)
import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_type
from bodo.hiframes.datetime_date_ext import DatetimeDateType
from bodo.transforms import distributed_pass, distributed_analysis
from bodo.utils.utils import debug_prints
from bodo.transforms.distributed_analysis import Distribution
from bodo.libs.str_ext import string_type
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.bool_arr_ext import boolean_array
from bodo.utils.utils import sanitize_varname
from bodo import objmode
from bodo.hiframes.pd_categorical_ext import PDCategoricalDtype, CategoricalArray


def connector_array_analysis(node, equiv_set, typemap, array_analysis):
    post = []
    # empty csv/parquet/sql/json nodes should be deleted in remove dead
    assert len(node.out_vars) > 0, "empty {} in array analysis".format(
        node.connector_typ
    )

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variable for an output column
    all_shapes = []
    for col_var in node.out_vars:
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


def connector_distributed_analysis(node, array_dists):
    # all output arrays should have the same distribution
    out_dist = Distribution.OneD
    for v in node.out_vars:
        if v.name in array_dists:
            out_dist = Distribution(min(out_dist.value, array_dists[v.name].value))

    for v in node.out_vars:
        array_dists[v.name] = out_dist


def connector_typeinfer(node, typeinferer):
    for col_var, typ in zip(node.out_vars, node.out_types):
        typeinferer.lock_type(col_var.name, typ, loc=node.loc)
    return


def visit_vars_connector(node, callback, cbdata):
    if debug_prints():  # pragma: no cover
        print("visiting {} vars for:".format(node.connector_typ), node)
        print("cbdata: ", sorted(cbdata.items()))

    # update output_vars
    new_out_vars = []
    for col_var in node.out_vars:
        new_var = visit_vars_inner(col_var, callback, cbdata)
        new_out_vars.append(new_var)

    node.out_vars = new_out_vars
    if node.connector_typ in ("csv", "parquet", "json"):
        node.file_name = visit_vars_inner(node.file_name, callback, cbdata)
    return


def connector_usedefs(node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # output columns are defined
    def_set.update({v.name for v in node.out_vars})
    if node.connector_typ in ("csv", "parquet", "json"):
        use_set.add(node.file_name.name)

    return numba.core.analysis._use_defs_result(usemap=use_set, defmap=def_set)


def get_copies_connector(node, typemap):
    # csv/parquet/sql/json doesn't generate copies,
    # it just kills the output columns
    kill_set = set(v.name for v in node.out_vars)
    return set(), kill_set


def apply_copies_connector(
    node, var_dict, name_var_table, typemap, calltypes, save_copies
):
    """apply copy propagate in csv/parquet/sql/json"""

    # update output_vars
    new_out_vars = []
    for col_var in node.out_vars:
        new_var = replace_vars_inner(col_var, var_dict)
        new_out_vars.append(new_var)

    node.out_vars = new_out_vars
    if node.connector_typ in ("csv", "parquet", "json"):
        node.file_name = replace_vars_inner(node.file_name, var_dict)
    return


def build_connector_definitions(node, definitions=None):
    if definitions is None:
        definitions = defaultdict(list)

    for col_var in node.out_vars:
        definitions[col_var.name].append(node)

    return definitions


class StreamReaderType(types.Opaque):
    def __init__(self):
        super(StreamReaderType, self).__init__(name="StreamReaderType")


stream_reader_type = StreamReaderType()
register_model(StreamReaderType)(models.OpaqueModel)


@box(StreamReaderType)
def box_stream_reader(typ, val, c):
    return val
