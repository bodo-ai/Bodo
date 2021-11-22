# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Common IR extension functions for connectors such as CSV, Parquet and JSON readers.
"""
from collections import defaultdict

import numba
from numba.core import ir, types
from numba.core.ir_utils import replace_vars_inner, visit_vars_inner
from numba.extending import box, models, register_model

from bodo.hiframes.table import TableType
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
    # for table types, out_vars consists of the table and the index value,
    # which should also the same length in the first dimension
    all_shapes = []

    for col_var in node.out_vars:
        typ = typemap[col_var.name]
        # parquet node's index variable may be None if there is no index array
        if typ == types.none:
            continue
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

    # For non Table returns, all output arrays should have the same distribution
    # For Table returns, both the table and the index should have the same distriution
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

    # new table format case
    if node.connector_typ == "csv":
        if node.chunksize is not None:
            # Iterator is stored in out types
            typeinferer.lock_type(
                node.out_vars[0].name, node.out_types[0], loc=node.loc
            )
        else:
            typeinferer.lock_type(
                node.out_vars[0].name, TableType(tuple(node.out_types)), loc=node.loc
            )
            typeinferer.lock_type(
                node.out_vars[1].name, node.index_column_typ, loc=node.loc
            )
        return

    if node.connector_typ == "parquet":
        typeinferer.lock_type(
            node.out_vars[0].name, TableType(tuple(node.out_types)), loc=node.loc
        )
        typeinferer.lock_type(
            node.out_vars[1].name, node.index_column_type, loc=node.loc
        )
        return

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

    if node.connector_typ in ("parquet", "sql") and node.filters:
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

    if node.connector_typ in ("parquet", "sql") and node.filters:
        for predicate_list in node.filters:
            for v in predicate_list:
                # If v[2] is a variable add it to the use set. If its
                # a compile time constant (i.e. NULL) then don't add it.
                if isinstance(v[2], ir.Var):
                    use_set.add(v[2].name)

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
    if node.connector_typ in ("parquet", "sql") and node.filters:
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
        defs = definitions[col_var.name]
        # In certain compiler passes, like typing_pass and series_pass,
        # we remove definitions for assignments. However, we don't do this
        # for Bodo Custom IR nodes, which makes certain function (like
        # get_definitions) fail if the definition is added multiple times.
        # As a result, we don't add the definition if it already present.
        # TODO: Remove the IR nodes whenever we remove definitions for assignments.
        if node not in defs:
            defs.append(node)

    return definitions


def generate_filter_map(filters):
    """
    Function used by connectors with filter pushdown. Givens filters, which are
    either a list of filters in arrow format or None, it returns a dictionary
    mapping ir.Var.name -> runtime_name and a list of unique ir.Vars.
    """
    if filters:
        filter_vars = []
        # handle predicate pushdown variables that need to be passed to C++/SQL
        pred_vars = [v[2] for predicate_list in filters for v in predicate_list]
        # variables may be repeated due to distribution of Or over And in predicates, so
        # remove duplicates. Cannot use ir.Var objects in set directly.
        var_set = set()
        for var in pred_vars:
            if isinstance(var, ir.Var):
                if var.name not in var_set:
                    filter_vars.append(var)
                var_set.add(var.name)
        return {v.name: f"f{i}" for i, v in enumerate(filter_vars)}, filter_vars
    else:
        return {}, []


class StreamReaderType(types.Opaque):
    def __init__(self):
        super(StreamReaderType, self).__init__(name="StreamReaderType")


stream_reader_type = StreamReaderType()
register_model(StreamReaderType)(models.OpaqueModel)


@box(StreamReaderType)
def box_stream_reader(typ, val, c):
    c.pyapi.incref(val)
    return val


def trim_extra_used_columns(used_columns, num_columns):
    """
    Trim a computed set of used columns to eliminate any columns
    beyond the num_columns available at the source. This is necessary
    because a set_table_data call could introduce new columns which
    would be initially included to load (see test_table_extra_column)

    used_columns is assumed to be a sorted list in increasing order.
    """
    end = len(used_columns)
    for i in range(len(used_columns) - 1, -1, -1):
        # Columns are sorted in reverse order, so iterate backwards
        # until we find a column within the original table bounds
        if used_columns[i] < num_columns:
            break
        end = i
    return used_columns[:end]
