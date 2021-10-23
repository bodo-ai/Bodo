# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Common IR extension functions for connectors such as CSV, Parquet and JSON readers.
"""
from collections import defaultdict

import numba
from numba.core import types
from numba.core.ir_utils import replace_vars_inner, visit_vars_inner
from numba.extending import box, models, register_model

from bodo.transforms.distributed_analysis import Distribution
from bodo.utils.utils import debug_prints


def connector_array_analysis(node, equiv_set, typemap, array_analysis):
    post = []
    # empty csv/parquet/sql/json nodes should be deleted in remove dead
    assert len(node.out_vars) > 0, "empty {} in array analysis".format(
        node.connector_typ
    )

    # If we have a csv chunksize the variables don't refer to the data,
    # so we skip this step.
    if node.connector_typ == "csv" and node.chunksize is not None:
        return [], []
    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variable for an output column
    all_shapes = []
    for col_var in node.out_vars:
        typ = typemap[col_var.name]
        shape = array_analysis._gen_shape_call(equiv_set, col_var, typ.ndim, None, post)
        equiv_set.insert_equiv(col_var, shape)
        all_shapes.append(shape[0])
        equiv_set.define(col_var, set())

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post


def connector_distributed_analysis(node, array_dists):
    """
    Common distributed analysis function shared by
    various connectors.
    """
    # Import inside function to avoid circular import
    from bodo.ir.sql_ext import SqlReader

    # If we have a SQL node with an inferred limit, we may have a
    # 1D-Var distribution
    if isinstance(node, SqlReader) and node.limit is not None:
        out_dist = Distribution.OneD_Var
    else:
        out_dist = Distribution.OneD
    # all output arrays should have the same distribution
    for v in node.out_vars:
        if v.name in array_dists:
            out_dist = Distribution(min(out_dist.value, array_dists[v.name].value))

    for v in node.out_vars:
        array_dists[v.name] = out_dist


def connector_typeinfer(node, typeinferer):
    """
    Set the typing constraints for various connector nodes.
    This is used for showing type dependencies. As a result,
    connectors only require that the output columns exactly
    match the types expected from read_csv.

    While the inputs fields of these nodes have type requirements,
    these should only be checked after the typemap is finalized
    because they should not allow the inputs to unify at all.
    """
    for col_var, typ in zip(node.out_vars, node.out_types):
        typeinferer.lock_type(col_var.name, typ, loc=node.loc)


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

    if node.connector_typ == "csv":
        node.nrows = visit_vars_inner(node.nrows, callback, cbdata)
        node.skiprows = visit_vars_inner(node.skiprows, callback, cbdata)

    if node.connector_typ == "parquet" and node.filters:
        for predicate in node.filters:
            for i in range(len(predicate)):
                val = predicate[i]
                # e.g. handle ("A", "==", v)
                predicate[i] = (
                    val[0],
                    val[1],
                    visit_vars_inner(val[2], callback, cbdata),
                )


def connector_usedefs(node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # output columns are defined
    def_set.update({v.name for v in node.out_vars})
    if node.connector_typ in ("csv", "parquet", "json"):
        use_set.add(node.file_name.name)

    if node.connector_typ == "csv":
        # Default value of nrows=-1, skiprows=0
        if isinstance(node.nrows, numba.core.ir.Var):
            use_set.add(node.nrows.name)
        if isinstance(node.skiprows, numba.core.ir.Var):
            use_set.add(node.skiprows.name)

    if node.connector_typ == "parquet" and node.filters:
        use_set.update(
            {v[2].name for predicate_list in node.filters for v in predicate_list}
        )

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
    if node.connector_typ == "parquet" and node.filters:
        for predicate in node.filters:
            for i in range(len(predicate)):
                val = predicate[i]
                # e.g. handle ("A", "==", v)
                predicate[i] = (val[0], val[1], replace_vars_inner(val[2], var_dict))
    if node.connector_typ == "csv":
        node.nrows = replace_vars_inner(node.nrows, var_dict)
        node.skiprows = replace_vars_inner(node.skiprows, var_dict)


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
    c.pyapi.incref(val)
    return val
