"""
Common IR extension functions for connectors such as CSV, Parquet and JSON readers.
"""

import sys
import typing as pt
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import llvmlite.binding as ll
import numba
import pandas as pd
from numba.core import ir, types
from numba.core.ir_utils import replace_vars_inner, visit_vars_inner

import bodo
import bodo.ir.filter as bif
from bodo.hiframes.table import TableType
from bodo.io import arrow_cpp  # type: ignore
from bodo.io.arrow_reader import ArrowReaderType
from bodo.ir.filter import (
    Filter,
    FilterVisitor,
    Scalar,
    supported_arrow_funcs_map,
)
from bodo.transforms.distributed_analysis import Distribution
from bodo.transforms.table_column_del_pass import get_live_column_nums_block
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.typing import BodoError
from bodo.utils.utils import (
    debug_prints,
    is_array_typ,
)

if pt.TYPE_CHECKING:  # pragma: no cover
    from numba.core.typeinfer import TypeInferer

    from bodo.ir.filter import Op, Ref

ll.add_symbol("arrow_reader_read_py_entry", arrow_cpp.arrow_reader_read_py_entry)


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
    chunksize: int | None = None

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
    if node.connector_typ in ("csv", "parquet", "sql", "iceberg") and node.is_streaming:
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
            and node.connector_typ in ("parquet", "sql", "iceberg")
            and not node.is_live_table
        )
        # If its the file_list or snapshot ID don't generate the shape
        is_non_array = node.connector_typ in ("sql", "iceberg") and i > 1
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


class VarVisitor(FilterVisitor[bif.Filter]):
    """
    Traverse the Filter expression tree. For every Scalar node, apply
    a transformation function to the internal ir.Var.

    Args:
        func (Callable): The function to apply to each variable.

    Returns:
        The transformed Bodo IR filter expression.
    """

    func: pt.Callable[[ir.Var], ir.Var]

    def __init__(self, func: pt.Callable[[ir.Var], ir.Var]):
        self.func = func

    def visit_scalar(self, scalar: "Scalar") -> "Filter":
        return bif.Scalar(self.func(scalar.val))

    def visit_ref(self, ref: "Ref") -> "Filter":
        return ref

    def visit_op(self, filter: "Op") -> "Filter":
        return bif.Op(filter.op, *[self.visit(arg) for arg in filter.args])


def visit_vars_connector(node: Connector, callback, cbdata):
    if debug_prints():  # pragma: no cover
        print(f"visiting {node.connector_typ} vars for:", node)
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

    if node.connector_typ in ("parquet", "sql", "iceberg") and node.filters:
        visitor = VarVisitor(lambda v: visit_vars_inner(v, callback, cbdata))
        node.filters = visitor.visit(node.filters)

    if node.connector_typ in ("sql", "iceberg") and node.rtjf_terms:
        for i in range(len(node.rtjf_terms)):
            var, indices, non_equality_info = node.rtjf_terms[i]
            new_var = visit_vars_inner(var, callback, cbdata)
            node.rtjf_terms[i] = (new_var, indices, non_equality_info)

    if node.connector_typ == "iceberg":
        node.connection = visit_vars_inner(node.connection, callback, cbdata)


def get_filter_vars(filters: Filter) -> list[ir.Var]:
    """
    get all variables in filters of a read node (that will be pushed down)

    Args:
        filters (Filter): filters (list of predicates)

    Returns:
        list(ir.Var): all variables in filters
    """
    filter_vars: list[ir.Var] = []

    def _append_var(v):
        filter_vars.append(v)
        return v

    var_visitor = VarVisitor(_append_var)
    var_visitor.visit(filters)
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

    if node.connector_typ in ("parquet", "sql", "iceberg") and node.filters:
        vars = get_filter_vars(node.filters)
        use_set.update({v.name for v in vars})

    if node.connector_typ in ("sql", "iceberg") and node.rtjf_terms:
        for i in range(len(node.rtjf_terms)):
            var, _, _ = node.rtjf_terms[i]
            if isinstance(var, numba.core.ir.Var):
                use_set.add(var.name)

    if node.connector_typ == "iceberg":
        if isinstance(node.connection, numba.core.ir.Var):
            use_set.add(node.connection.name)

    return numba.core.analysis._use_defs_result(usemap=use_set, defmap=def_set)


def get_copies_connector(node: Connector, typemap):
    # csv/parquet/sql/json doesn't generate copies,
    # it just kills the output columns
    kill_set = {v.name for v in node.out_vars}
    return set(), kill_set


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
    if node.connector_typ in ("parquet", "sql", "iceberg") and node.filters:
        visitor = VarVisitor(lambda v: replace_vars_inner(v, var_dict))
        node.filters = visitor.visit(node.filters)

    if node.connector_typ in ("sql", "iceberg") and node.rtjf_terms:
        for i in range(len(node.rtjf_terms)):
            var, indices, non_equality_info = node.rtjf_terms[i]
            new_var = replace_vars_inner(var, var_dict)
            node.rtjf_terms[i] = (new_var, indices, non_equality_info)

    if node.connector_typ == "iceberg":
        node.connection = replace_vars_inner(node.connection, var_dict)

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
StreamReaderType, _ = install_py_obj_class(
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
    assert isinstance(typemap[table_var_name], (TableType, ArrowReaderType)), (
        f"{nodename} Node Table must be a TableType or ArrowReaderMetaType"
    )

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

                node.out_used_cols = sorted(used_columns)
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
    assert node.is_streaming, (
        f"is_chunked_connector_table_parallel: {node_name} must be a connector in streaming mode"
    )

    parallel = False
    if array_dists is not None:
        iterator_varname = node.out_vars[0].name
        parallel = array_dists[iterator_varname] in (
            Distribution.OneD,
            Distribution.OneD_Var,
        )
    return parallel


VisitorOut = tuple[str, types.Type]


class ArrowFilterVisitor(FilterVisitor[VisitorOut]):
    """
    Convert Bodo IR filter expressions to string representation of Arrow Compute Expression.
    Also, determine if any casts are needed for the filter expression.

    Args:
        filter_map (dict[str, str]): Mapping of the IR variable name to the runtime variable name.
        original_out_types (tuple): A tuple of column data types for the input DataFrame, including dead
            columns.
        typemap (dict[str, types.Type]): Mapping of ir Variable names to their type.
        orig_colname_map (dict[str, int]): Mapping of column name to its column index.
        partition_names (list[str]): List of column names that represent parquet partitions.
        source (Literal["parquet", "iceberg"]): The input source that needs the filters.
            Either parquet or iceberg.
        output_f_string (bool): Whether the expression filter should be returned as an f-string
            where the column names are templated instead of being inlined. This is used for
            Iceberg to allow us to generate the expression dynamically for different file
            schemas to account for schema evolution.

    Returns:
        str: String representation of the Arrow Compute Expression.
        types.Type: The type of the Arrow Compute Expression. Expected to be bool by end
    """

    def __init__(
        self,
        filter_map: dict[str, str],
        original_out_types: tuple[types.Type, ...],
        typemap,
        orig_colname_map: dict[str, int],
        partition_names,
        source: pt.Literal["parquet", "iceberg"],
        output_f_string: bool = False,
    ):
        self.filter_map = filter_map
        self.original_out_types = original_out_types
        self.typemap = typemap
        self.orig_colname_map = orig_colname_map
        self.partition_names = partition_names
        self.source = source
        self.output_f_string = output_f_string

    def unwrap_scalar(self, scalar: Filter) -> VisitorOut:
        if not isinstance(scalar, Scalar):
            raise ValueError("ArrowFilterVisitor::unwrap_scalar: bif.Scalar expected.")
        return self.filter_map[scalar.val.name], self.typemap[scalar.val.name]

    def determine_filter_cast(self, lhs_array_type, rhs_typ) -> tuple[str, str]:
        return determine_filter_cast(
            lhs_array_type,
            rhs_typ,  # self.partition_names, self.source
        )

    def visit_scalar(self, scalar: "Scalar") -> VisitorOut:
        """Convert Scalar Values to Arrow Compute Expression String"""
        fname, ftype = self.unwrap_scalar(scalar)
        return f"ds.scalar({fname})", ftype

    def visit_ref(self, ref: "Ref") -> VisitorOut:
        col_type = self.original_out_types[self.orig_colname_map[ref.val]]

        if self.source == "parquet" and ref.val in self.partition_names:
            # Always cast partitions to protect again multiple types
            # with parquet (see test_read_partitions_string_int).
            # We skip this with Iceberg because partitions are hidden.
            if col_type == types.unicode_type:
                col_cast = ".cast(pa.string(), safe=False)"
            elif isinstance(col_type, types.Integer):
                # all arrow types integer type names are the same as numba
                # type names.
                col_cast = f".cast(pa.{col_type.name}(), safe=False)"
            else:
                # Currently arrow support int and string partitions, so we only capture those casts
                # https://github.com/apache/arrow/blob/230afef57f0ccc2135ced23093bac4298d5ba9e4/python/pyarrow/parquet.py#L989
                col_cast = ""
        else:
            col_cast = ""

        ref_str = f"{{{ref.val}}}" if self.output_f_string else ref.val
        return f"ds.field('{ref_str}'){col_cast}", col_type

    def visit_op(self, filter: "Op") -> VisitorOut:
        match filter.op:
            case "ALWAYS_TRUE":
                return "ds.scalar(True)", types.bool_
            case "ALWAYS_FALSE":
                return "ds.scalar(False)", types.bool_
            case "ALWAYS_NULL":
                return "ds.scalar(None)", types.bool_

            # Logical Operators
            case "NOT":
                return f"~({self.visit(filter.args[0])[0]})", types.bool_
            case "AND":
                return (
                    "(" + " & ".join(self.visit(arg)[0] for arg in filter.args) + ")",
                    types.bool_,
                )
            case "OR":
                return (
                    "(" + " | ".join(self.visit(arg)[0] for arg in filter.args) + ")",
                    types.bool_,
                )

            # Unary Operators Specially Handled
            case "IS_NULL":
                return f"({self.visit(filter.args[0])[0]}.is_null())", types.bool_
            case "IS_NOT_NULL":
                return f"(~{self.visit(filter.args[0])[0]}.is_null())", types.bool_

            # Binary Operators with Special Handling
            case "IN":
                col_code, col_type = self.visit(filter.args[0])
                scalar_code, scalar_type = self.unwrap_scalar(filter.args[1])
                col_cast, scalar_cast = determine_filter_cast(col_type, scalar_type)

                # col_cast, scalar_cast = self.determine_filter_cast(
                # Expected output for this format should look like
                # ds.field('A').isin(filter_var)
                return (
                    f"({col_code}{col_cast}.isin({scalar_code}{scalar_cast}))",
                    types.bool_,
                )
            case "case_insensitive_equality":
                col_code, col_type = self.visit(filter.args[0])
                scalar_code, scalar_type = self.unwrap_scalar(filter.args[1])
                col_cast, scalar_cast = determine_filter_cast(col_type, scalar_type)

                # case_insensitive_equality is just
                # == with both inputs converted to lower case. This is used
                # by ilike
                return (
                    f"(pa.compute.ascii_lower({col_code}{col_cast}) == pa.compute.ascii_lower(ds.scalar({scalar_code}{scalar_cast}))",
                    types.bool_,
                )

            case "COALESCE":
                col_code, col_type = self.visit(filter.args[0])
                scalars = [self.visit(f) for f in filter.args[1:]]
                col_cast, scalar_cast = determine_filter_cast(col_type, scalars[0][1])

                scalar_codes = (f"{s[0]}{scalar_cast}" for s in scalars)
                return (
                    f"pa.compute.coalesce({col_code}{col_cast}, {', '.join(scalar_codes)})",
                    col_type,
                )

            # Comparison Operators Syntax
            case "==" | "!=" | "<" | "<=" | ">" | ">=":
                col_code, col_type = self.visit(filter.args[0])
                scalar_code, scalar_type = self.visit(filter.args[1])
                col_cast, scalar_cast = determine_filter_cast(col_type, scalar_type)

                # Expected output for this format should like
                # (ds.field('A') > ds.scalar(py_var))
                return (
                    f"({col_code}{col_cast} {filter.op} {scalar_code}{scalar_cast})",
                    types.bool_,
                )

            # All Other Arrow Functions
            case op:
                func_name = supported_arrow_funcs_map[op.lower()]
                col_expr, col_type = self.visit(filter.args[0])
                scalar_args = [self.unwrap_scalar(f)[0] for f in filter.args[1:]]

                # Handle if its case insensitive
                all_args = [col_expr] + scalar_args
                if op.startswith("case_insensitive_"):
                    return (
                        f"(pa.compute.{func_name}({', '.join(all_args)}, ignore_case=True))",
                        col_type,
                    )
                else:
                    return (
                        f"(pa.compute.{func_name}({', '.join(all_args)}))",
                        col_type,
                    )


def generate_arrow_filters(
    filters: Filter | None,
    filter_map,
    col_names,
    partition_names,
    original_out_types,
    typemap,
    source: pt.Literal["parquet", "iceberg"],
    output_expr_filters_as_f_string=False,
    sql_op_id: int = -1,
) -> str:
    """
    Generate Arrow expression filters with the given filter_map.
    Construct an ArrowFilterVisitor to do so

    Keyword arguments:
    filters -- DNF expression from the IR node for filters. None
               if there are no filters.
    filter_map -- Mapping from filter value to var name.
    col_names -- original column names in the IR node, including dead columns.
    partition_names -- Column names that can be used as partitions.
    original_out_types -- original column types in the IR node, including dead columns.
    typemap -- Maps variables name -> types.
    source -- What is generating this filter. Either "parquet" or "iceberg".
    output_expr_filters_as_f_string -- Whether to output the expression filter
        as an f-string, where the column names are templated. This is used in Iceberg
        for schema evolution purposes to allow substituting the column names
        used in the filter based on the file/schema-group. See description
        of bodo.io.iceberg.generate_expr_filter for more details.
    sql_op_id -- Operator ID generated by the planner for the TableScan operator associated
        with this filter. This is only used in the generated verbose log message.
        If the value is -1 (the default), the operator ID will not be included in the log message.
    """
    expr_filter_str = "None"
    # If no filters use variables (i.e. all isna, then we still need to take this path)
    if filters:
        # Create a mapping for faster column indexing
        orig_colname_map = {c: i for i, c in enumerate(col_names)}

        arrow_filter_visitor = ArrowFilterVisitor(
            filter_map,
            original_out_types,
            typemap,
            orig_colname_map,
            partition_names,
            source,
            output_expr_filters_as_f_string,
        )
        expr_filter_str, _ = arrow_filter_visitor.visit(filters)

    if bodo.user_logging.get_verbose_level() >= 1:
        op_id_msg = f" (Operator ID: {sql_op_id}) " if sql_op_id != -1 else ""
        msg = "Arrow filters pushed down%s:\n%s\n"
        bodo.user_logging.log_message(
            "Filter Pushdown",
            msg,
            op_id_msg,
            expr_filter_str,
        )

    return expr_filter_str


def determine_filter_cast(
    lhs_array_type: types.ArrayCompatible,
    rhs_typ,
) -> tuple[str, str]:
    """
    Function that generates text for casts that need to be included
    in the filter when not automatically handled by Arrow. For example
    timestamp and string. This function returns two strings. In most cases
    one of the strings will be empty and the other contain the argument that
    should be cast. However, if we have a partition column, we always cast the
    partition column either to its original type or the new type.

    We opt to cast in the direction that keep maximum information, for
    example date -> timestamp rather than timestamp -> date.

    Args:
        lhs_array_type -- Type of the original column.
        rhs_typ -- Type of the filter scalar.

    Returns:
        - A string that contains the cast for the column.
        - A string that contains the cast for the filter scalar.
    """

    lhs_scalar_typ = bodo.utils.typing.element_type(lhs_array_type)
    col_cast = ""

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
            return ".cast(pa.timestamp('ns'), safe=False)", ""
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
            return col_cast, ".cast(pa.timestamp('ns'), safe=False)"
        elif lhs_scalar_typ == bodo.datetime_date_type and rhs_scalar_typ in (
            bodo.datetime64ns,
            bodo.pd_timestamp_tz_naive_type,
        ):
            return ".cast(pa.timestamp('ns'), safe=False)", ""
        elif rhs_scalar_typ == bodo.datetime_date_type and lhs_scalar_typ in (
            bodo.datetime64ns,
            bodo.pd_timestamp_tz_naive_type,
        ):  # pragma: no cover
            return col_cast, ".cast(pa.timestamp('ns'), safe=False)"
    return col_cast, ""


def log_limit_pushdown(io_node: Connector, read_size: int):
    """Log that either Bodo or BodoSQL has performed limit pushdown.
    This may not capture all limit pushdown to Snowflake from BodoSQL, but will
    capture limit pushdown to Iceberg from BodoSQL.

    Args:
        io_node (Connector): The connector used for logging.
        read_size (int): The constant number of rows to read. If/When we support
            non-constant limits, this will need to be updated.
    """
    if bodo.user_logging.get_verbose_level() >= 1:
        if io_node.connector_typ == "sql":
            node_name = f"{io_node.db_type} sql node"
        else:
            node_name = f"{io_node.connector_typ} node"
        msg = f"Successfully performed limit pushdown on {node_name}: %s\n %s\n"
        io_source = io_node.loc.strformat()
        constant_limit_message = (
            f"Constant limit detected, reading at most {read_size} rows"
        )
        bodo.user_logging.log_message(
            "Limit Pushdown", msg, io_source, constant_limit_message
        )
