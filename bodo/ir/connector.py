# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Common IR extension functions for connectors such as CSV, Parquet and JSON readers.
"""
from collections import defaultdict
from typing import Literal, Tuple

import numba
from numba.core import ir, types
from numba.core.ir_utils import replace_vars_inner, visit_vars_inner
from numba.extending import box, models, register_model

from bodo.hiframes.table import TableType
from bodo.transforms.distributed_analysis import Distribution
from bodo.transforms.table_column_del_pass import get_live_column_nums_block
from bodo.utils.typing import BodoError
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
    if isinstance(node, SqlReader) and not node.is_select_query:
        out_dist = Distribution.REP
    elif isinstance(node, SqlReader) and node.limit is not None:
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

    if node.connector_typ in ("parquet", "sql"):
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


def cast_float_to_nullable(df, df_type):
    """
    Takes a DataFrame read in objmode and casts
    columns that are only floats due to null values as
    Nullable integers.
    https://stackoverflow.com/questions/57960179/how-do-i-prevent-nulls-from-causing-the-wrong-datatype-in-a-dataframe
    """
    import bodo

    col_map = {}
    for i, coltype in enumerate(df_type.data):
        if isinstance(coltype, bodo.IntegerArrayType):
            dtype = coltype.get_pandas_scalar_type_instance
            if dtype not in col_map:
                col_map[dtype] = []
            col_map[dtype].append(df.columns[i])
    for typ, cols in col_map.items():
        df[cols] = df[cols].astype(typ)


def connector_table_column_use(node, block_use_map, equiv_vars, typemap):
    """
    Function to handle any necessary processing for column uses
    with a particular table. This is used for connectors that define
    a table and don't use any other table, so this does nothing.

    This is currently used by:
        CSVReader
        ParquetReader
        SQLReader
    """
    return


def base_connector_remove_dead_columns(
    node, column_live_map, equiv_vars, typemap, nodename, possible_cols
):
    """
    Function that tracks which columns to prune from a connector IR node.
    This updates out_used_cols which stores which arrays in the
    types will need to actually be loaded.

    This is mapped to the used columns during distributed pass.
    """
    # The function assumes nodes have exactly 2 vars
    assert len(node.out_vars) == 2, f"invalid {nodename} node"
    table_var_name = node.out_vars[0].name
    assert isinstance(
        typemap[table_var_name], TableType
    ), f"{nodename} Node Table must be a TableType"
    # if possible_cols == [] then the table is dead and we are only loading
    # the index. See 'remove_dead_sql' or 'remove_dead_pq' for examples.
    if possible_cols:
        # Compute all columns that are live at this statement.
        used_columns, use_all = get_live_column_nums_block(
            column_live_map, equiv_vars, table_var_name
        )
        used_columns = trim_extra_used_columns(used_columns, len(possible_cols))
        if not use_all and not used_columns:
            # If we see no specific column is need some operations need some
            # column but no specific column. For example:
            # T = read_parquet(table(0, 1, 2, 3))
            # n = len(T)
            #
            # Here we just load column 0. If no columns are actually needed, dead
            # code elimination will remove the entire IR var in 'remove_dead_parquet'.
            #
            used_columns = [0]
        if not use_all and len(used_columns) != len(node.out_used_cols):
            # Update the type offset. If an index column its not included in
            # the original table. If we have code like
            #
            # T = read_csv(table(0, 1, 2, 3)) # Assume index column is column 2
            #
            # We type T without the index column as Table(arr0, arr1, arr3).
            # As a result once we apply optimizations, all the column indices
            # will refer to the index within that type, not the original file.
            #
            # i.e. T[2] == arr3
            #
            # This means that used_columns will track the offsets within the type,
            # not the actual column numbers in the file. We keep these offsets separate
            # while finalizing DCE and we will update the file with the actual columns later
            # in distirbuted pass.
            #
            # For more information see:
            # https://bodo.atlassian.net/wiki/spaces/B/pages/921042953/Table+Structure+with+Dead+Columns#User-Provided-Column-Pruning-at-the-Source

            node.out_used_cols = used_columns
            # Return that this table was updated

    """We return flase in all cases, as no changes performed in the file will allow for dead code elimination to do work."""
    return False


def is_connector_table_parallel(node, array_dists, typemap, node_name):
    """
    Returns if the parallel implementation should be used for
    a connector that returns two variables, a table and an
    index.
    """
    parallel = False
    if array_dists is not None:
        # table is parallel
        table_varname = node.out_vars[0].name
        parallel = array_dists[table_varname] in (
            Distribution.OneD,
            Distribution.OneD_Var,
        )
        index_varname = node.out_vars[1].name
        # index array parallelism should match the table
        assert (
            typemap[index_varname] == types.none
            or not parallel
            or array_dists[index_varname]
            in (
                Distribution.OneD,
                Distribution.OneD_Var,
            )
        ), f"{node_name} data/index parallelization does not match"
    return parallel


def generate_arrow_filters(
    filters,
    filter_map,
    filter_vars,
    col_names,
    partition_names,
    original_out_types,
    typemap,
    source: Literal["parquet", "iceberg"],
) -> Tuple[str, str]:
    """
    Generate Arrow DNF filters and expression filters with the
    given filter_map and filter_vars.

    Keyword arguments:
    filters -- DNF expression from the IR node for filters. None
               if there are no filters.
    filter_map -- Mapping from filter value to var name.
    filter_vars -- List of filter vars
    col_names -- original column names in the IR node, including dead columns.
    partition_names -- Column names that can be used as partitions.
    original_out_types -- original column types in the IR node, including dead columns.
    typemap -- Maps variables name -> types.
    source -- What is generating this filter. Either "parquet" or "iceberg".
    """
    dnf_filter_str = "None"
    expr_filter_str = "None"
    # If no filters use variables (i.e. all isna, then we still need to take this path)
    if filters:
        # Create two formats for parquet/arrow. One using the old DNF format
        # which contains just the partition columns (partition pushdown)
        # and one using the more verbose expression format and all expressions
        # (predicate pushdown).
        # https://arrow.apache.org/docs/python/dataset.html#filtering-data
        #
        # partition pushdown requires the expressions to just contain Hive partitions
        # If any expressions are not hive partitions then the DNF filters should treat
        # them as true
        #
        # For example if A, C are parition column names and B is not
        #
        # Then for partition expressions:
        # ((A > 4) & (B < 2)) | ((A < 2) & (C == 1))
        # => ((A > 4) & True) | ((A < 2) & (C == 1))
        # => (A > 4) | ((A < 2) & (C == 1))
        #
        # Similarly if any OR expression consist of all True we do not have
        # any partition filters.
        #
        # For example f A, C are parition column names and B is not
        # (B < 2) | ((A < 2) & (C == 1))
        # => True | ((A < 2) & (C == 1))
        # => True
        #
        # So we set dnf_filter_str = "None"
        #
        # expr_filter_str always contains all of the filters, regardless of if
        # the column is a partition column.
        #
        dnf_or_conds = []
        expr_or_conds = []
        # If any filters aren't on a partition column or are unsupported
        # in partitions, then this expression must effectively be replaced with TRUE.
        # If any of the AND sections contain only TRUE expresssions, then we must won't
        # be able to filter anything based on partitioning. To represent this we will set
        # skip_partitions = True, which will set  dnf_filter_str = "None".
        skip_partitions = False
        # Create a mapping for faster column indexing
        orig_colname_map = {c: i for i, c in enumerate(col_names)}
        for predicate in filters:
            dnf_and_conds = []
            expr_and_conds = []
            for v in predicate:
                # First update expr since these include all expressions.
                # If v[2] is a var we pass the variable at runtime,
                # otherwise we pass a constant (i.e. NULL) which
                # requires special handling
                if isinstance(v[2], ir.Var):
                    # expr conds don't do some of the casts that DNF expressions do (for example String and Timestamp).
                    # For known cases where we need to cast we generate the cast. For simpler code we return two strings,
                    # column_cast and scalar_cast.
                    column_cast, scalar_cast = determine_filter_cast(
                        original_out_types,
                        typemap,
                        v,
                        orig_colname_map,
                        partition_names,
                        source,
                    )
                    if v[1] == "in":
                        # Expected output for this format should like
                        # (ds.field('A').isin(py_var))
                        expr_and_conds.append(
                            f"(ds.field('{v[0]}').isin({filter_map[v[2].name]}))"
                        )
                    else:
                        # Expected output for this format should like
                        # (ds.field('A') > ds.scalar(py_var))
                        expr_and_conds.append(
                            f"(ds.field('{v[0]}'){column_cast} {v[1]} ds.scalar({filter_map[v[2].name]}){scalar_cast})"
                        )
                else:
                    # Currently the only constant expressions we support are IS [NOT] NULL
                    assert v[2] == "NULL", "unsupport constant used in filter pushdown"
                    if v[1] == "is not":
                        prefix = "~"
                    else:
                        prefix = ""
                    # Expected output for this format should like
                    # (~ds.field('A').is_null())
                    expr_and_conds.append(f"({prefix}ds.field('{v[0]}').is_null())")
                # Now handle the dnf section. We can only append a value if its not a constant
                # expression and is a partition column. If we already know skip_partitions = False,
                # then we skip partitions as they will be unused.

                if not skip_partitions:
                    if v[0] in partition_names and isinstance(v[2], ir.Var):
                        dnf_str = f"('{v[0]}', '{v[1]}', {filter_map[v[2].name]})"
                        dnf_and_conds.append(dnf_str)
                    # handle isna/notna (e.g. [[('C', 'is', 'NULL')]]) cases only for
                    # Iceberg, since supporting nulls in Parquet/Arrow/Hive partitioning is
                    # complicated (e.g. Spark allows users to specify custom null directory)
                    elif (
                        v[0] in partition_names
                        and not isinstance(v[2], ir.Var)
                        and source == "iceberg"
                    ):
                        dnf_str = f"('{v[0]}', '{v[1]}', '{v[2]}')"
                        dnf_and_conds.append(dnf_str)
                    # If we don't append to the list, we are effectively
                    # replacing this expression with TRUE as
                    # (expr AND TRUE => expr)

            dnf_and_str = ""
            if dnf_and_conds:
                dnf_and_str = ", ".join(dnf_and_conds)
            else:
                # If dnf_and_conds is empty, this expression = TRUE
                # Since (expr OR TRUE => TRUE), we should omit all partitions.
                skip_partitions = True
            expr_and_str = " & ".join(expr_and_conds)
            # If all the filters are truncated we may have an empty string.
            if dnf_and_str:
                dnf_or_conds.append(f"[{dnf_and_str}]")
            expr_or_conds.append(f"({expr_and_str})")

        dnf_or_str = ", ".join(dnf_or_conds)
        expr_or_str = " | ".join(expr_or_conds)
        if dnf_or_str and not skip_partitions:
            # If the expression exists use and we don't need
            # to skip partitions we need  update dnf_filter_str
            dnf_filter_str = f"[{dnf_or_str}]"
        expr_filter_str = f"({expr_or_str})"
    return dnf_filter_str, expr_filter_str


def determine_filter_cast(
    col_types, typemap, filter_val, orig_colname_map, partition_names, source
):
    """
    Function that generates text for casts that need to be included
    in the filter when not automatically handled by Arrow. For example
    timestamp and string. This function returns two strings. In most cases
    one of the strings will be empty and the other contain the argument that
    should be cast. However, if we have a partition column, we always cast the
    partition column either to its original type or the new type.

    This is because of an issue in arrow where if a partion has a mix of integer
    and string names it won't look at the global types when Bodo processes per
    file (see test_read_partitions_string_int for an example).

    We opt to cast in the direction that keep maximum information, for
    example date -> timestamp rather than timestamp -> date.

    Right now we assume filter_val[0] is a column name and filter_val[2]
    is a scalar Var that is found in the typemap.

    Keyword arguments:
    col_types -- Types of the original columns, including dead columns.
    typemap -- Maps variables name -> types.
    filter_val -- Filter value DNF expression
    orig_colname_map -- Map index -> column name.
    partition_names -- List of column names that can be used as partitions.
    source -- What is generating this filter. Either "parquet" or "iceberg".
    """
    import bodo

    colname = filter_val[0]
    lhs_arr_typ = col_types[orig_colname_map[colname]]
    lhs_scalar_typ = bodo.utils.typing.element_type(lhs_arr_typ)
    if source == "parquet" and colname in partition_names:
        # Always cast partitions to protect again multiple types
        # with parquet (see test_read_partitions_string_int).
        # We skip this with Iceberg because partitions are hidden.

        if lhs_scalar_typ == types.unicode_type:
            col_cast = ".cast(pyarrow.string(), safe=False)"
        elif isinstance(lhs_scalar_typ, types.Integer):
            # all arrow types integer type names are the same as numba
            # type names.
            col_cast = f".cast(pyarrow.{lhs_scalar_typ.name}(), safe=False)"
        else:
            # Currently arrow support int and string partitions, so we only capture those casts
            # https://github.com/apache/arrow/blob/230afef57f0ccc2135ced23093bac4298d5ba9e4/python/pyarrow/parquet.py#L989
            col_cast = ""
    else:
        col_cast = ""
    rhs_typ = typemap[filter_val[2].name]
    # If we do isin, then rhs_typ will be a list or set
    if isinstance(rhs_typ, (types.List, types.Set)):
        rhs_scalar_typ = rhs_typ.dtype
    else:
        rhs_scalar_typ = rhs_typ

    # Here we assume is_common_scalar_dtype conversions are common
    # enough that Arrow will support them, since these are conversions
    # like int -> float. TODO: Test
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(
        lhs_scalar_typ, "Filter pushdown"
    )
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(
        rhs_scalar_typ, "Filter pushdown"
    )
    if not bodo.utils.typing.is_common_scalar_dtype([lhs_scalar_typ, rhs_scalar_typ]):
        # If a cast is not implicit it must be in our white list.
        if not bodo.utils.typing.is_safe_arrow_cast(lhs_scalar_typ, rhs_scalar_typ):
            raise BodoError(
                f"Unsupported Arrow cast from {lhs_scalar_typ} to {rhs_scalar_typ} in filter pushdown. Please try a comparison that avoids casting the column."
            )
        # We always cast string -> other types
        # Only supported types should be string and timestamp
        if lhs_scalar_typ == types.unicode_type:
            return ".cast(pyarrow.timestamp('ns'), safe=False)", ""
        elif lhs_scalar_typ in (bodo.datetime64ns, bodo.pd_timestamp_type):
            if isinstance(rhs_typ, (types.List, types.Set)):  # pragma: no cover
                # This path should never be reached because we checked that
                # list/set doesn't contain Timestamp or datetime64 in typing pass.
                type_name = "list" if isinstance(rhs_typ, types.List) else "tuple"
                raise BodoError(
                    f"Cannot cast {type_name} values with isin filter pushdown."
                )
            return col_cast, ".cast(pyarrow.timestamp('ns'), safe=False)"

    return col_cast, ""
