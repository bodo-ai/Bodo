# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Common IR extension functions for connectors such as CSV, Parquet and JSON readers.
"""
import sys
import typing as pt
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numba
import pandas as pd
from numba.core import ir, types
from numba.core.ir_utils import replace_vars_inner, visit_vars_inner

import bodo
from bodo.hiframes.table import TableType
from bodo.io.arrow_reader import ArrowReaderType
from bodo.ir.filter import Filter, supported_arrow_funcs_map
from bodo.transforms.distributed_analysis import Distribution
from bodo.transforms.table_column_del_pass import get_live_column_nums_block
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.typing import BodoError
from bodo.utils.utils import (
    debug_prints,
    get_filter_predicate_compute_func,
    is_array_typ,
)

if pt.TYPE_CHECKING:  # pragma: no cover
    from numba.core.typeinfer import TypeInferer


class Connector(ir.Stmt, metaclass=ABCMeta):
    connector_typ: str

    # Numba IR Properties
    loc: ir.Loc
    out_vars: list[ir.Var]
    # Original out var, for debugging only
    df_out_varname: str

    # Output Dataframe / Table Typing
    out_table_col_names: list[str]
    out_table_col_types: list[types.ArrayCompatible]

    # Is Streaming Enabled, and Whats the Output Table Size
    chunksize: pt.Optional[int] = None

    @property
    def is_streaming(self) -> bool:
        """Will the Connector Output a Single Table Batch or a Stream"""
        return self.chunksize is not None

    @abstractmethod
    def out_vars_and_types(self) -> list[tuple[str, types.Type]]:
        """
        Returns the output variables and their types. Used in the
        default implementation of Connector.typeinfer_out_vars
        """
        ...

    def typeinfer_out_vars(self, typeinferer: "TypeInferer") -> None:
        """
        Set the typing constraints for the current connector node.
        This is used for showing type dependencies. As a result,
        connectors only require that the output columns exactly
        match the types expected.

        While the inputs fields of these nodes have type requirements,
        these should only be checked after the typemap is finalized
        because they should not allow the inputs to unify at all.
        """
        for var, typ in self.out_vars_and_types():
            typeinferer.lock_type(var, typ, loc=self.loc)

    def out_table_distribution(self) -> Distribution:
        return Distribution.OneD


def connector_array_analysis(node: Connector, equiv_set, typemap, array_analysis):
    post = []
    # empty csv/parquet/sql/json nodes should be deleted in remove dead
    assert len(node.out_vars) > 0, f"Empty {node.connector_typ} in Array Analysis"

    # If we have a csv chunksize the variables don't refer to the data,
    # so we skip this step.
    if node.connector_typ in ("csv", "parquet", "sql") and node.is_streaming:
        return [], []

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variable for an output column
    # for table types, out_vars consists of the table and the index value,
    # which should also the same length in the first dimension
    all_shapes = []

    for i, col_var in enumerate(node.out_vars):
        typ = typemap[col_var.name]
        # parquet node's index variable may be None if there is no index array
        if typ == types.none:
            continue
        # If the table variable is dead don't generate the shape call.
        is_dead_table = (
            i == 0
            and node.connector_typ in ("parquet", "sql")
            and not node.is_live_table
        )
        # If its the file_list or snapshot ID don't generate the shape
        is_non_array = node.connector_typ == "sql" and i > 1
        if not (is_dead_table or is_non_array):
            shape = array_analysis._gen_shape_call(
                equiv_set, col_var, typ.ndim, None, post
            )
            equiv_set.insert_equiv(col_var, shape)
            all_shapes.append(shape[0])
            equiv_set.define(col_var, set())

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post


def connector_distributed_analysis(node: Connector, array_dists):
    """
    Common distributed analysis function shared by
    various connectors.
    """
    out_dist = node.out_table_distribution()

    # For non Table returns, all output arrays should have the same distribution
    # For Table returns, both the table and the index should have the same distribution
    for v in node.out_vars:
        if v.name in array_dists:
            out_dist = Distribution(min(out_dist.value, array_dists[v.name].value))

    for v in node.out_vars:
        array_dists[v.name] = out_dist


def connector_typeinfer(node: Connector, typeinferer: "TypeInferer") -> None:
    """
    Set the typing constraints for various connector nodes.
    See Connector.typeinfer_out_vars for more information.
    """

    node.typeinfer_out_vars(typeinferer)


def _visit_predicate_tuple_vars(tup, callback, cbdata):
    """visit variables in a tuple representing a filter pushdown predicate
    e.g. ('l_commitdate', 'coalesce', v1)

    Args:
        tup (tuple): tuple representing predicate
        callback (function): variable visit callback function
        cbdata (any): callback function's input

    Returns:
        tuple: updated tuple values
    """
    assert isinstance(tup, tuple), "_visit_predicate_tuple_vars: tuple input expected"

    out_data = []
    for v in tup:
        out_val = v
        if isinstance(v, tuple):
            out_val = _visit_predicate_tuple_vars(v, callback, cbdata)
        elif isinstance(v, ir.Var):
            out_val = visit_vars_inner(v, callback, cbdata)

        out_data.append(out_val)

    return tuple(out_data)


def visit_vars_connector(node: Connector, callback, cbdata):
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
                val = list(predicate[i])
                # e.g. handle ("A", "==", v),
                # [(('l_commitdate', 'coalesce', v1), '>=', v2)]
                val[0] = (
                    _visit_predicate_tuple_vars(val[0], callback, cbdata)
                    if isinstance(val[0], tuple)
                    else val[0]
                )
                predicate[i] = (
                    val[0],
                    val[1],
                    visit_vars_inner(val[2], callback, cbdata),
                )


def _get_predicate_tuple_vars(tup):
    """get variables in a tuple representing a filter pushdown predicate
    (could be nested)
    e.g. ('l_commitdate', 'coalesce', v1) -> [v1]

    Args:
        tup (tuple): tuple of predicate

    Returns:
        list(ir.Var): variables in predicate
    """
    assert isinstance(tup, tuple), "_get_predicate_tuple_vars: tuple input expected"
    vars = []
    for v in tup:
        if isinstance(v, ir.Var):
            vars.append(v)
        if isinstance(v, tuple):
            vars += _get_predicate_tuple_vars(v)

    return vars


def get_filter_vars(filters):
    """get all variables in filters of a read node (that will be pushed down)
    e.g. [[(('l_commitdate', 'coalesce', v1), '>=', v2)]] -> [v1, v2]

    Args:
        filters (list(list)): filters (list of predicates)

    Returns:
        list(ir.Var): all variables in filters
    """
    filter_vars = []
    for predicate_list in filters:
        for v in predicate_list:
            filter_vars += _get_predicate_tuple_vars(v)

    return filter_vars


def connector_usedefs(node: Connector, use_set=None, def_set=None):
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
        vars = get_filter_vars(node.filters)
        use_set.update({v.name for v in vars})

    return numba.core.analysis._use_defs_result(usemap=use_set, defmap=def_set)


def get_copies_connector(node: Connector, typemap):
    # csv/parquet/sql/json doesn't generate copies,
    # it just kills the output columns
    kill_set = set(v.name for v in node.out_vars)
    return set(), kill_set


def _replace_predicate_tuple_vars(tup, var_dict):
    """replace variables in a tuple representing a filter pushdown predicate
    e.g. ('l_commitdate', 'coalesce', v1)

    Args:
        tup (tuple): tuple representing predicate
        var_dict (dict(str, ir.Var)): map for variable replacement

    Returns:
        tuple: updated tuple values
    """
    assert isinstance(tup, tuple), "_replace_predicate_tuple_vars: tuple input expected"

    out_data = []
    for v in tup:
        out_val = v
        if isinstance(v, tuple):
            out_val = _replace_predicate_tuple_vars(v, var_dict)
        elif isinstance(v, ir.Var):
            out_val = replace_vars_inner(v, var_dict)

        out_data.append(out_val)

    return tuple(out_data)


def apply_copies_connector(
    node: Connector, var_dict, name_var_table, typemap, calltypes, save_copies
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
                val = list(predicate[i])
                # e.g. handle ("A", "==", v),
                # [(('l_commitdate', 'coalesce', v1), '>=', v2)]
                val[0] = (
                    _replace_predicate_tuple_vars(val[0], var_dict)
                    if isinstance(val[0], tuple)
                    else val[0]
                )
                predicate[i] = (val[0], val[1], replace_vars_inner(val[2], var_dict))
    if node.connector_typ == "csv":
        node.nrows = replace_vars_inner(node.nrows, var_dict)
        node.skiprows = replace_vars_inner(node.skiprows, var_dict)


def build_connector_definitions(node: Connector, definitions=None):
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
        pred_vars = get_filter_vars(filters)
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


this_module = sys.modules[__name__]
StreamReaderType = install_py_obj_class(
    types_name="stream_reader_type",
    module=this_module,
    class_name="StreamReaderType",
    model_name="StreamReaderModel",
)


def trim_extra_used_columns(used_columns: set[int], num_columns: int) -> set[int]:
    """
    Trim a computed set of used columns to eliminate any columns
    beyond the num_columns available at the source. This is necessary
    because a set_table_data call could introduce new columns which
    would be initially included to load (see test_table_extra_column)


    Args:
        used_columns (set): Set of used columns
        num_columns (int): Total number of possible columns.
            All columns >= num_columns should be removed.

    Returns:
        Set: Set of used columns after removing any out of
            bounds columns.
    """
    return {i for i in used_columns if i < num_columns}


def cast_float_to_nullable(df, df_type):
    """
    Takes a DataFrame read in objmode and casts
    columns that are only floats due to null values as
    Nullable integers.
    https://stackoverflow.com/questions/57960179/how-do-i-prevent-nulls-from-causing-the-wrong-datatype-in-a-dataframe
    """
    import bodo

    col_map = defaultdict(list)
    for i, coltype in enumerate(df_type.data):
        if isinstance(coltype, (bodo.IntegerArrayType, bodo.FloatingArrayType)):
            dtype = coltype.get_pandas_scalar_type_instance
            col_map[dtype].append(df.columns[i])
    for typ, cols in col_map.items():
        # Pandas (as of 1.4) may create an object array for nullable float types with
        # nulls as 'NaN' string values. Converting to Numpy first avoids failure in
        # astype(). See test_s3_read_json
        if isinstance(typ, (pd.Float32Dtype, pd.Float64Dtype)):
            df[cols] = df[cols].astype(typ.numpy_dtype).astype(typ)
        else:
            df[cols] = df[cols].astype(typ)


def connector_table_column_use(
    node: Connector, block_use_map, equiv_vars, typemap, table_col_use_map
):
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
    node: Connector,
    column_live_map,
    equiv_vars,
    typemap,
    nodename,
    possible_cols,
    require_one_column=True,
):
    """
    Function that tracks which columns to prune from a connector IR node.
    This updates out_used_cols which stores which arrays in the
    types will need to actually be loaded.

    This is mapped to the used columns during distributed pass.
    """
    table_var_name = node.out_vars[0].name
    table_key = (table_var_name, None)

    # Arrow reader is equivalent to tables for column elimination purposes
    assert isinstance(
        typemap[table_var_name], (TableType, ArrowReaderType)
    ), f"{nodename} Node Table must be a TableType or ArrowReaderMetaType"

    # if possible_cols == [] then the table is dead and we are only loading
    # the index. See 'remove_dead_sql' or 'remove_dead_pq' for examples.
    if possible_cols:
        # Compute all columns that are live at this statement.
        used_columns, use_all, cannot_del_cols = get_live_column_nums_block(
            column_live_map, equiv_vars, table_key
        )
        if not (use_all or cannot_del_cols):
            used_columns = trim_extra_used_columns(used_columns, len(possible_cols))
            if not used_columns and require_one_column:
                # If we see no specific column is need some operations need some
                # column but no specific column. For example:
                # T = read_parquet(table(0, 1, 2, 3))
                # n = len(T)
                #
                # Here we just load column 0. If no columns are actually needed, dead
                # code elimination will remove the entire IR var in 'remove_dead_parquet'.
                #
                used_columns = {0}
            if len(used_columns) != len(node.out_used_cols):
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
                # in distributed pass.
                #
                # For more information see:
                # https://bodo.atlassian.net/wiki/spaces/B/pages/921042953/Table+Structure+with+Dead+Columns#User-Provided-Column-Pruning-at-the-Source

                node.out_used_cols = list(sorted(used_columns))
                # Return that this table was updated

    # We return false in all cases, as no changes performed
    # in the file will allow for dead code elimination to do work.
    return False


def is_connector_table_parallel(
    node: Connector, array_dists, typemap, node_name
) -> bool:
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


def is_chunked_connector_table_parallel(node, array_dists, node_name):
    """
    Returns if the parallel implementation should be used for
    a connector that returns an iterator
    """
    assert (
        node.is_streaming
    ), f"is_chunked_connector_table_parallel: {node_name} must be a connector in streaming mode"

    parallel = False
    if array_dists is not None:
        iterator_varname = node.out_vars[0].name
        parallel = array_dists[iterator_varname] in (
            Distribution.OneD,
            Distribution.OneD_Var,
        )
    return parallel


def _get_filter_column_arrow_expr(col_val, filter_map, output_f_string: bool = False):
    """returns Arrow expr code for filter column,
    e.g. ("A", "coalesce", v1) - > pa.compute.coalesce(ds.field('A'), ds.scalar(f1))

    Args:
        col_val (tuple|str): column representation in filter
        filter_map (dict(str, int)): map of IR variable names to read function variable
            names. E.g. {'_v14call_method_6_224': 'f0'}
        output_f_string (bool): Whether we should return an f-string where the
            column names are templated variables instead of being inlined.

    Returns:
        str: Arrow expression for filter column
    """
    if isinstance(col_val, str):
        return (
            f"ds.field('{{{col_val}}}')"
            if output_f_string
            else f"ds.field('{col_val}')"
        )

    filter = _get_filter_column_arrow_expr(col_val[0], filter_map, output_f_string)
    column_compute_func = get_filter_predicate_compute_func(col_val)

    if column_compute_func == "coalesce":
        scalar_val = (
            filter_map[col_val[2].name]
            if isinstance(col_val[2], ir.Var)
            else col_val[2]
        )
        return f"pa.compute.coalesce({filter}, ds.scalar({scalar_val}))"

    elif column_compute_func in supported_arrow_funcs_map:  # pragma: no cover
        arrow_func_name = supported_arrow_funcs_map[column_compute_func]
        return f"pa.compute.{arrow_func_name}({filter})"


def generate_arrow_filters(
    filters,
    filter_map,
    filter_vars,
    col_names,
    partition_names,
    original_out_types,
    typemap,
    source: pt.Literal["parquet", "iceberg"],
    output_dnf=True,
    output_expr_filters_as_f_string=False,
) -> tuple[str, str]:
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
    output_dnf -- Should we output the first expression in DNF format or regular
                  arrow compute expression format.
    output_expr_filters_as_f_string -- Whether to output the expression filter
        as an f-string, where the column names are templated. This is used in Iceberg
        for schema evolution purposes to allow substituting the column names
        used in the filter based on the file/schema-group. See description
        of bodo.io.iceberg.generate_expr_filter for more details.
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
        # If any of the AND sections contain only TRUE expressions, then we must won't
        # be able to filter anything based on partitioning. To represent this we will set
        # skip_partitions = True, which will set  dnf_filter_str = "None".
        skip_partitions = False
        # Create a mapping for faster column indexing
        orig_colname_map = {c: i for i, c in enumerate(col_names)}
        for predicate in filters:
            dnf_and_conds = []
            expr_and_conds = []
            for p in predicate:
                # First update expr since these include all expressions.
                # If p[2] is a var we pass the variable at runtime,
                # otherwise we pass a constant (i.e. NULL) which
                # requires special handling
                expr_val = _generate_column_expr_filter(
                    p,
                    filter_map,
                    original_out_types,
                    typemap,
                    orig_colname_map,
                    partition_names,
                    source,
                    output_expr_filters_as_f_string,
                )
                expr_and_conds.append(expr_val)
                # Now handle the dnf section. We can only append a value if its not a constant
                # expression and is a partition column. If we already know skip_partitions = False,
                # then we skip partitions as they will be unused.

                if not skip_partitions:
                    # operators supported in DNF expressions. is/is not is only supported by iceberg
                    dnf_ops = {
                        "==",
                        "!=",
                        "<",
                        ">",
                        "<=",
                        ">=",
                        "in",
                        "is",
                        "is not",
                    }
                    if (
                        p[0] in partition_names
                        and isinstance(p[2], ir.Var)
                        and p[1] in dnf_ops
                    ):
                        if output_dnf:
                            dnf_str = f"('{p[0]}', '{p[1]}', {filter_map[p[2].name]})"
                        else:
                            dnf_str = (
                                expr_val
                                if not output_expr_filters_as_f_string
                                # If the expr_val is an f-string, add the
                                # column names to it for the dnf_str.
                                else expr_val.format(**{x: x for x in col_names})
                            )
                        dnf_and_conds.append(dnf_str)
                    # handle isna/notna (e.g. [[('C', 'is', 'NULL')]]) cases only for
                    # Iceberg, since supporting nulls in Parquet/Arrow/Hive partitioning is
                    # complicated (e.g. Spark allows users to specify custom null directory)
                    elif (
                        p[0] in partition_names
                        and not isinstance(p[2], ir.Var)
                        and source == "iceberg"
                        and p[1] in dnf_ops
                    ):
                        if output_dnf:
                            dnf_str = f"('{p[0]}', '{p[1]}', '{p[2]}')"
                        else:
                            dnf_str = (
                                expr_val
                                if not output_expr_filters_as_f_string
                                # If the expr_val is an f-string, add the
                                # column names to it for the dnf_str.
                                else expr_val.format(**{x: x for x in col_names})
                            )
                        dnf_and_conds.append(dnf_str)
                    # If we don't append to the list, we are effectively
                    # replacing this expression with TRUE as
                    # (expr AND TRUE => expr)

            dnf_and_str = ""
            if dnf_and_conds:
                if output_dnf:
                    dnf_and_str = ", ".join(dnf_and_conds)
                else:
                    dnf_and_str = " & ".join(dnf_and_conds)
            else:
                # If dnf_and_conds is empty, this expression = TRUE
                # Since (expr OR TRUE => TRUE), we should omit all partitions.
                skip_partitions = True
            expr_and_str = " & ".join(expr_and_conds)
            # If all the filters are truncated we may have an empty string.
            if dnf_and_str:
                if output_dnf:
                    dnf_or_conds.append(f"[{dnf_and_str}]")
                else:
                    dnf_or_conds.append(f"({dnf_and_str})")
            expr_or_conds.append(f"({expr_and_str})")

        if output_dnf:
            dnf_or_str = ", ".join(dnf_or_conds)
        else:
            dnf_or_str = " | ".join(dnf_or_conds)
        expr_or_str = " | ".join(expr_or_conds)
        if dnf_or_str and not skip_partitions:
            # If the expression exists use and we don't need
            # to skip partitions we need  update dnf_filter_str
            if output_dnf:
                dnf_filter_str = f"[{dnf_or_str}]"
            else:
                dnf_filter_str = f"({dnf_or_str})"
        expr_filter_str = f"({expr_or_str})"

    if bodo.user_logging.get_verbose_level() >= 1:
        msg = "Arrow filters pushed down:\n%s\n%s\n"
        bodo.user_logging.log_message(
            "Filter Pushdown",
            msg,
            dnf_filter_str,
            expr_filter_str,
        )

    return dnf_filter_str, expr_filter_str


def _get_filter_column_type(
    col_val: pt.Union[str, Filter],
    col_types: pt.Sequence[types.Type],
    orig_colname_map: dict[str, int],
):
    """get column type for column representation in filter predicate

    Args:
        col_val: column name or compute representation
        col_types: output data types of read node
        orig_colname_map: map of column name to its index in output

    Returns:
        types.Type: data type of column
    """
    if isinstance(col_val, str):
        return col_types[orig_colname_map[col_val]]

    func_name = get_filter_predicate_compute_func(col_val)
    if func_name == "length":
        # Length converts the column type to integer.
        return bodo.IntegerArrayType(types.int64)
    return _get_filter_column_type(col_val[0], col_types, orig_colname_map)


def determine_filter_cast(
    col_types: pt.Sequence[types.Type],
    typemap,
    filter_val: list[pt.Union[str, Filter]],
    orig_colname_map: dict[str, int],
    partition_names,
    source: str,
) -> tuple[str, str]:
    """
    Function that generates text for casts that need to be included
    in the filter when not automatically handled by Arrow. For example
    timestamp and string. This function returns two strings. In most cases
    one of the strings will be empty and the other contain the argument that
    should be cast. However, if we have a partition column, we always cast the
    partition column either to its original type or the new type.

    This is because of an issue in arrow where if a partition has a mix of integer
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
    orig_colname_map -- Map column name -> index
    partition_names -- List of column names that can be used as partitions.
    source -- What is generating this filter. Either "parquet" or "iceberg".
    """
    import bodo

    colname = filter_val[0]
    lhs_arr_typ = _get_filter_column_type(filter_val[0], col_types, orig_colname_map)
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
    # If we do series isin, then rhs_typ will be a list or set
    if isinstance(rhs_typ, (types.List, types.Set)):
        rhs_scalar_typ = rhs_typ.dtype
    # If we do isin via the bodosql array kernel, then rhs_typ will be an array
    # We enforce that this array is replicated, so it's safe to do pushdown
    elif is_array_typ(rhs_typ):
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
        # All paths are only tested via slow except date -> timestamp
        if not bodo.utils.typing.is_safe_arrow_cast(
            lhs_scalar_typ, rhs_scalar_typ
        ):  # pragma: no cover
            raise BodoError(
                f"Unsupported Arrow cast from {lhs_scalar_typ} to {rhs_scalar_typ} in filter pushdown. Please try a comparison that avoids casting the column."
            )
        # We always cast string -> other types
        # Only supported types should be string and timestamp or timestamp + date
        if lhs_scalar_typ == types.unicode_type and rhs_scalar_typ in (
            bodo.datetime64ns,
            bodo.pd_timestamp_tz_naive_type,
        ):  # pragma: no cover
            return ".cast(pyarrow.timestamp('ns'), safe=False)", ""
        elif rhs_scalar_typ == types.unicode_type and lhs_scalar_typ in (
            bodo.datetime64ns,
            bodo.pd_timestamp_tz_naive_type,
        ):  # pragma: no cover
            if isinstance(rhs_typ, (types.List, types.Set)):  # pragma: no cover
                # This path should never be reached because we checked that
                # list/set doesn't contain Timestamp or datetime64 in typing pass.
                type_name = "list" if isinstance(rhs_typ, types.List) else "tuple"
                raise BodoError(
                    f"Cannot cast {type_name} values with isin filter pushdown."
                )
            return col_cast, ".cast(pyarrow.timestamp('ns'), safe=False)"
        elif lhs_scalar_typ == bodo.datetime_date_type and rhs_scalar_typ in (
            bodo.datetime64ns,
            bodo.pd_timestamp_tz_naive_type,
        ):
            return ".cast(pyarrow.timestamp('ns'), safe=False)", ""
        elif rhs_scalar_typ == bodo.datetime_date_type and lhs_scalar_typ in (
            bodo.datetime64ns,
            bodo.pd_timestamp_tz_naive_type,
        ):  # pragma: no cover
            return col_cast, ".cast(pyarrow.timestamp('ns'), safe=False)"
    return col_cast, ""


def _generate_column_expr_filter(
    filter: Filter,
    filter_map: dict[str, str],
    original_out_types: tuple,
    typemap: dict[str, types.Type],
    orig_colname_map: dict[str, int],
    partition_names: list[str],
    source: pt.Literal["parquet", "iceberg"],
    output_f_string: bool = False,
) -> str:
    """Generates an Arrow format expression filter representing the comparison for a single column.

    Args:
        filter (tuple[Union[str, tuple], str, Union[ir.Var, str]]): The column filter to parse.
        filter_map (dict[str, str]): Mapping of the IR variable name to the runtime variable name.
        original_out_types (tuple): A tuple of column data types for the input DataFrame, including dead
            columns.
        typemap (dict[str, types.Type]): Mapping of ir Variable names to their type.
        orig_colname_map (dict[int, str]): Mapping of column index to its column name.
        partition_names (list[str]): List of column names that represent parquet partitions.
        source (Literal["parquet", "iceberg"]): The input source that needs the filters.
            Either parquet or iceberg.
        output_f_string (bool): Whether the expression filter should be returned as an f-string
            where the column names are templated instead of being inlined. This is used for
            Iceberg to allow us to generate the expression dynamically for different file
            schemas to account for schema evolution.

    Returns:
        str: A string representation of an arrow expression equivalent to the filter.
    """
    p0, p1, p2 = filter
    if p1 == "ALWAYS_TRUE":
        # Special operator for True.
        expr_val = "ds.scalar(True)"
    elif p1 == "ALWAYS_FALSE":
        # Special operator for False
        expr_val = "ds.scalar(False)"
    elif p1 == "ALWAYS_NULL":
        # Special operator for NULL
        expr_val = "ds.scalar(None)"
    elif p1 == "not":
        inner_filter = _generate_column_expr_filter(
            p0,
            filter_map,
            original_out_types,
            typemap,
            orig_colname_map,
            partition_names,
            source,
            output_f_string,
        )
        expr_val = f"~({inner_filter})"
    else:
        col_expr = _get_filter_column_arrow_expr(p0, filter_map, output_f_string)
        if isinstance(p2, ir.Var):
            # expr conds don't do some of the casts that DNF expressions do (for example String and Timestamp).
            # For known cases where we need to cast we generate the cast. For simpler code we return two strings,
            # column_cast and scalar_cast.
            column_cast, scalar_cast = determine_filter_cast(
                original_out_types,
                typemap,
                filter,
                orig_colname_map,
                partition_names,
                source,
            )
            filter_var = filter_map[p2.name]
            if p1 == "in":
                # Expected output for this format should look like
                # ds.field('A').isin(filter_var)
                expr_val = f"({col_expr}.isin({filter_var}))"
            elif p1 == "case_insensitive_equality":
                # case_insensitive_equality is just
                # == with both inputs converted to lower case. This is used
                # by ilike
                expr_val = f"(pa.compute.ascii_lower({col_expr}{column_cast}) == pa.compute.ascii_lower(ds.scalar({filter_var}){scalar_cast}))"

            elif p1 in supported_arrow_funcs_map:  # pragma: no cover
                op = p1
                func_name = supported_arrow_funcs_map[p1]
                scalar_arg = filter_var

                # Handle if its case insensitive
                if op.startswith("case_insensitive_"):
                    expr_val = f"(pa.compute.{func_name}({col_expr}, {scalar_arg}, ignore_case=True))"
                else:
                    expr_val = f"(pa.compute.{func_name}({col_expr}, {scalar_arg}))"
            else:
                # Expected output for this format should like
                # (ds.field('A') > ds.scalar(py_var))
                expr_val = f"({col_expr}{column_cast} {p1} ds.scalar({filter_var}){scalar_cast})"
        else:
            # Currently the only constant expressions we support are IS [NOT] NULL
            assert p2 == "NULL", "unsupported constant used in filter pushdown"
            if p1 == "is not":
                prefix = "~"
            else:
                prefix = ""
            # Expected output for this format should like
            # (~ds.field('A').is_null())
            expr_val = f"({prefix}{col_expr}.is_null())"
    return expr_val
