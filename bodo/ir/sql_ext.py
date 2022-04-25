# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Implementation of pd.read_sql in BODO.
We piggyback on the pandas implementation. Future plan is to have a faster
version for this task.
"""

import numba
import numpy as np
import pandas as pd
from numba.core import ir, ir_utils, typeinfer, types
from numba.core.ir_utils import (
    compile_to_numba_ir,
    next_label,
    replace_arg_nodes,
)

import bodo
import bodo.ir.connector
from bodo import objmode
from bodo.hiframes.table import Table, TableType
from bodo.libs.array import (
    cpp_table_to_py_table,
    delete_table,
    info_from_table,
    info_to_array,
    table_type,
)
from bodo.libs.distributed_api import bcast, bcast_scalar
from bodo.libs.str_ext import string_type, unicode_to_utf8
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.transforms.table_column_del_pass import (
    ir_extension_table_column_use,
    remove_dead_column_extensions,
)
from bodo.utils.typing import BodoError
from bodo.utils.utils import check_and_propagate_cpp_exception

MPI_ROOT = 0


class SqlReader(ir.Stmt):
    def __init__(
        self,
        sql_request,
        connection,
        df_out,
        df_colnames,
        out_vars,
        out_types,
        converted_colnames,
        db_type,
        loc,
        unsupported_columns,
        unsupported_arrow_types,
        is_select_query,
        index_column_name,
        index_column_type,
    ):
        self.connector_typ = "sql"
        self.sql_request = sql_request
        self.connection = connection
        self.df_out = df_out  # used only for printing
        self.df_colnames = df_colnames
        self.out_vars = out_vars
        self.out_types = out_types
        # Any columns that had their output name converted by the actual
        # DB result. This is used by Snowflake because we update the SQL query
        # to perform dce and we must specify the exact column name (because we quote
        # escape the names). This may include both the table column names and the
        # index column.
        self.converted_colnames = converted_colnames
        self.loc = loc
        self.limit = req_limit(sql_request)
        self.db_type = db_type
        # Support for filter pushdown. Currently only used with snowflake.
        self.filters = None
        # These fields are used to enable compilation if unsupported columns
        # get eliminated. Currently only used with snowflake.
        self.unsupported_columns = unsupported_columns
        self.unsupported_arrow_types = unsupported_arrow_types
        self.is_select_query = is_select_query
        # Name of the index column. None if index=False.
        self.index_column_name = index_column_name
        # Type of the index array. types.none if index=False.
        self.index_column_type = index_column_type
        # List of indices within the table name that are used.
        # df_colnames is unchanged unless the table is deleted,
        # so this is used to track dead columns.
        self.type_usecol_offset = list(range(len(df_colnames)))

    def __repr__(self):  # pragma: no cover
        return f"{self.df_out} = ReadSql(sql_request={self.sql_request}, connection={self.connection}, col_names={self.df_colnames}, types={self.out_types}, vars={self.out_vars}, limit={self.limit}, unsupported_columns={self.unsupported_columns}, unsupported_arrow_types={self.unsupported_arrow_types}, is_select_query={self.is_select_query}, index_column_name={self.index_column_name}, index_column_type={self.index_column_type}, type_usecol_offset={self.type_usecol_offset},)"


def remove_dead_sql(
    sql_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):
    """
    Regular Dead Code elimination function for the SQLReader Node.
    The SQLReader node returns two IR variables (the table and the index).
    If neither of these variables is used after various dead code elimination
    in various compiler passes, the SQLReader node will be removed entirely
    (the return None path).

    However, its possible one of the IR variables may be eliminated but not
    the entire node. For example, if the index is unused then that IR variable
    may be dead, but the table is still used then we cannot eliminate the entire
    SQLReader node. In this case we must update the node internals to reflect
    that the single IR variable can be eliminated and won't be loaded in the
    SQL query.

    This does not include column elimination on the table.
    """
    table_var = sql_node.out_vars[0].name
    index_var = sql_node.out_vars[1].name
    if table_var not in lives and index_var not in lives:
        # If neither the table or index is live, remove the node.
        return None
    elif table_var not in lives:
        # If table isn't live we only want to load the index.
        # To do this we should mark the df_colnames as empty
        sql_node.out_types = []
        sql_node.df_colnames = []
        sql_node.type_usecol_offset = []
    elif index_var not in lives:
        # If the index_var not in lives we don't load the index.
        # To do this we mark the index_column_name as None
        sql_node.index_column_name = None
        sql_node.index_arr_typ = types.none
    return sql_node


def sql_distributed_run(
    sql_node, array_dists, typemap, calltypes, typingctx, targetctx
):
    # Add debug info about column pruning
    if bodo.user_logging.get_verbose_level() >= 1:
        pruning_msg = "Finish column pruning on read_sql node:\n%s\nColumns loaded %s\n"
        sql_cols = []
        dict_encoded_cols = []
        for i in sql_node.type_usecol_offset:
            colname = sql_node.df_colnames[i]
            sql_cols.append(colname)
            if isinstance(
                sql_node.out_types[i], bodo.libs.dict_arr_ext.DictionaryArrayType
            ):
                dict_encoded_cols.append(colname)
        # Include the index since it needs to be loaded from the query
        if sql_node.index_column_name:
            sql_cols.append(sql_node.index_column_name)
            if isinstance(
                sql_node.index_column_type, bodo.libs.dict_arr_ext.DictionaryArrayType
            ):
                dict_encoded_cols.append(sql_node.index_column_name)
        sql_source = sql_node.loc.strformat()
        bodo.user_logging.log_message(
            "Column Pruning",
            pruning_msg,
            sql_source,
            sql_cols,
        )
        # Log if any columns use dictionary encoded arrays.
        # TODO: Test. Dictionary encoding isn't supported yet.
        if dict_encoded_cols:
            encoding_msg = "Finished optimized encoding on read_sql node:\n%s\nColumns %s using dictionary encoding to reduce memory usage.\n"
            bodo.user_logging.log_message(
                "Dictionary Encoding",
                encoding_msg,
                sql_source,
                dict_encoded_cols,
            )

    parallel = bodo.ir.connector.is_connector_table_parallel(
        sql_node, array_dists, typemap, "SQLReader"
    )

    # Check for any unsupported columns still remaining
    if sql_node.unsupported_columns:
        # Determine the columns that were eliminated.
        unsupported_cols_set = set(sql_node.unsupported_columns)
        used_cols_set = set(sql_node.type_usecol_offset)
        # Compute the intersection of what was kept.
        remaining_unsupported = used_cols_set & unsupported_cols_set

        if remaining_unsupported:
            unsupported_list = sorted(remaining_unsupported)
            msg_list = [
                f"pandas.read_sql(): 1 or more columns found with Arrow types that are not supported in Bodo and could not be eliminated. "
                + "Please manually remove these columns from your sql query by specifying the columns you need in your SELECT statement. If these "
                + "columns are needed, you will need to modify your dataset to use a supported type.",
                "Unsupported Columns:",
            ]
            # Find the arrow types for the unsupported types
            idx = 0
            for col_num in unsupported_list:
                while sql_node.unsupported_columns[idx] != col_num:
                    idx += 1
                msg_list.append(
                    f"Column '{sql_node.original_df_colnames[col_num]}' with unsupported arrow type {sql_node.unsupported_arrow_types[idx]}"
                )
                idx += 1
            total_msg = "\n".join(msg_list)
            raise BodoError(total_msg, loc=sql_node.loc)

    filter_map, filter_vars = bodo.ir.connector.generate_filter_map(sql_node.filters)
    extra_args = ", ".join(filter_map.values())
    func_text = f"def sql_impl(sql_request, conn, {extra_args}):\n"
    if sql_node.filters:
        # If a predicate should be and together, they will be multiple tuples within the same list.
        # If predicates should be or together, they will be within separate lists.
        # i.e.
        # [[('l_linestatus', '<>', var1), ('l_shipmode', '=', var2))]]
        # -> -> (l_linestatus <> var1) AND (l_shipmode = var2)
        # [[('l_linestatus', '<>', var1)], [('l_shipmode', '=', var2))]]
        # -> (l_linestatus <> var1) OR (l_shipmode = var2)
        # [[('l_linestatus', '<>', var1)], [('l_shipmode', '=', var2))]]
        or_conds = []
        for and_list in sql_node.filters:
            and_conds = [
                # If p[2] is a constant that isn't in the IR (i.e. NULL)
                # just load the value directly, otherwise load the variable
                # at runtime.
                " ".join(
                    [
                        "(",
                        p[0],
                        p[1],
                        ("{" + filter_map[p[2].name] + "}")
                        if isinstance(p[2], ir.Var)
                        else p[2],
                        ")",
                    ]
                )
                for p in and_list
            ]
            or_conds.append(" ( " + " AND ".join(and_conds) + " ) ")
        where_cond = " WHERE " + " OR ".join(or_conds)
        for i, arg in enumerate(filter_map.values()):
            func_text += f"    {arg} = get_sql_literal({arg})\n"
        # Append filters via a format string. This format string is created and populated
        # at runtime because filter variables aren't necessarily constants (but they are scalars).
        func_text += f'    sql_request = f"{{sql_request}} {where_cond}"\n'
    func_text += "    (table_var, index_var) = _sql_reader_py(sql_request, conn)\n"
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    sql_impl = loc_vars["sql_impl"]

    sql_reader_py = _gen_sql_reader_py(
        sql_node.df_colnames,
        sql_node.out_types,
        sql_node.index_column_name,
        sql_node.index_column_type,
        sql_node.type_usecol_offset,
        typingctx,
        targetctx,
        sql_node.db_type,
        sql_node.limit,
        parallel,
        sql_node.is_select_query,
    )

    f_block = compile_to_numba_ir(
        sql_impl,
        {
            "_sql_reader_py": sql_reader_py,
            "bcast_scalar": bcast_scalar,
            "bcast": bcast,
            "get_sql_literal": _get_snowflake_sql_literal,
        },
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=(string_type, string_type)
        + tuple(typemap[v.name] for v in filter_vars),
        typemap=typemap,
        calltypes=calltypes,
    ).blocks.popitem()[1]

    if sql_node.is_select_query:
        # Prune the columns to only those that are used.
        used_col_names = [sql_node.df_colnames[i] for i in sql_node.type_usecol_offset]
        if sql_node.index_column_name:
            used_col_names.append(sql_node.index_column_name)
        # Update the SQL request to remove any unused columns. This is both
        # an optimization (the SQL engine loads less data) and is needed for
        # correctness. See test_sql_snowflake_single_column
        col_str = escape_column_names(
            used_col_names, sql_node.db_type, sql_node.converted_colnames
        )

        # https://stackoverflow.com/questions/33643163/in-oracle-as-alias-not-working
        if sql_node.db_type == "oracle":
            updated_sql_request = (
                "SELECT " + col_str + " FROM (" + sql_node.sql_request + ") TEMP"
            )
        else:
            updated_sql_request = (
                "SELECT " + col_str + " FROM (" + sql_node.sql_request + ") as TEMP"
            )
    else:
        updated_sql_request = sql_node.sql_request
    replace_arg_nodes(
        f_block,
        [
            ir.Const(updated_sql_request, sql_node.loc),
            ir.Const(sql_node.connection, sql_node.loc),
        ]
        + filter_vars,
    )
    nodes = f_block.body[:-3]
    # assign output table
    nodes[-2].target = sql_node.out_vars[0]
    # assign output index array
    nodes[-1].target = sql_node.out_vars[1]
    # At most one of the table and the index
    # can be dead because otherwise the whole
    # node should have already been removed.
    assert not (
        sql_node.index_column_name is None and not sql_node.type_usecol_offset
    ), "At most one of table and index should be dead if the SQL IR node is live"
    if sql_node.index_column_name is None:
        # If the index_col is dead, remove the node.
        nodes.pop(-1)
    elif not sql_node.type_usecol_offset:
        # If the table is dead, remove the node
        nodes.pop(-2)

    return nodes


def escape_column_names(col_names, db_type, converted_colnames):
    """
    Function that escapes column names when updating the SQL queries.
    Some outputs (i.e. count(*)) map to both functions and the output
    column names in certain dialects. If these are readded to the query,
    it may modify the results by rerunning the function, so we must
    escape the column names

    See: test_read_sql_column_function and test_sql_snowflake_count
    """
    # In Snowflake/Oracle we avoid functions by wrapping column names in quotes.
    # This makes the name case sensitive, so we avoid this by undoing any
    # conversions in the output as needed.
    if db_type in ("snowflake", "oracle"):
        # Snowflake/Oracle needs to convert all lower case strings back to uppercase
        used_col_names = [
            x.upper() if x in converted_colnames else x for x in col_names
        ]
        col_str = ", ".join([f'"{x}"' for x in used_col_names])

    # MySQL uses tilda as an escape character by default, not quotations
    # However, MySQL does support using quotations in ASCII_MODE. Tilda is always allowed though
    # MySQL names are case-insensitive
    elif db_type == "mysql" or db_type == "mysql+pymysql":
        col_str = ", ".join([f"`{x}`" for x in col_names])

    # By the SQL 1997 standard, wrapping with quotations should be the default
    # SQLite is the only DB tested with this functionality. SQLite column names are case-insensitive
    # MSSQL is good with or without quotes because it forces aliased subqueries to assign names to computed columns
    # For example, this is not allowed: SELECT * FROM (SELECT COUNT(*) from ___ GROUP BY ___)
    # But this is:                      SELECT * FROM (SELECT COUNT(*) as C from ___ GROUP BY ___)

    # PostgreSQL uses just the lowercased name of the function as the column name by default
    # E.x. SELECT COUNT(*) ... => Column Name is "count"
    # However columns can also always be escaped with quotes.
    # https://stackoverflow.com/questions/7651417/escaping-keyword-like-column-names-in-postgres
    else:
        col_str = ", ".join([f'"{x}"' for x in col_names])

    return col_str


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def _get_snowflake_sql_literal_scalar(filter_value):
    """
    Given a filter_value, which is a scalar python variable,
    returns a string representation of the filter value
    that could be used in a Snowflake SQL query.

    This is in a separate function to enable recursion.
    """
    filter_type = types.unliteral(filter_value)
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(
        filter_type, "Filter pushdown"
    )
    if filter_type == types.unicode_type:
        # Strings require double $$ to avoid escape characters
        # https://docs.snowflake.com/en/sql-reference/data-types-text.html#dollar-quoted-string-constants
        # TODO: Handle strings with $$ inside
        return lambda filter_value: f"$${filter_value}$$"  # pragma: no cover
    elif (
        isinstance(filter_type, (types.Integer, types.Float))
        or filter_value == types.bool_
    ):
        # Numeric and boolean values can just return the string representation
        return lambda filter_value: str(filter_value)  # pragma: no cover
    elif filter_type == bodo.pd_timestamp_type:
        # Timestamp needs to be converted to a timestamp literal
        def impl(filter_value):  # pragma: no cover
            nanosecond = filter_value.nanosecond
            nanosecond_prepend = ""
            if nanosecond < 10:
                nanosecond_prepend = "00"
            elif nanosecond < 100:
                nanosecond_prepend = "0"
            # TODO: Refactor once strftime support nanoseconds
            return f"timestamp '{filter_value.strftime('%Y-%m-%d %H:%M:%S.%f')}{nanosecond_prepend}{nanosecond}'"  # pragma: no cover

        return impl
    elif filter_type == bodo.datetime_date_type:
        # datetime.date needs to be converted to a date literal
        # Just return the string wrapped in quotes.
        # https://docs.snowflake.com/en/sql-reference/data-types-datetime.html#date
        return (
            lambda filter_value: f"date '{filter_value.strftime('%Y-%m-%d')}'"
        )  # pragma: no cover
    else:
        raise BodoError(
            f"pd.read_sql(): Internal error, unsupported scalar type {filter_type} used in filter pushdown."
        )


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def _get_snowflake_sql_literal(filter_value):
    """
    Given a filter_value, which is python variable,
    returns a string representation of the filter value
    that could be used in a Snowflake SQL query.
    """
    scalar_isinstance = (types.Integer, types.Float)
    scalar_equals = (
        bodo.datetime_date_type,
        bodo.pd_timestamp_type,
        types.unicode_type,
        types.bool_,
    )
    filter_type = types.unliteral(filter_value)
    if isinstance(filter_type, types.List) and (
        isinstance(filter_type.dtype, scalar_isinstance)
        or filter_type.dtype in scalar_equals
    ):
        # List are written as tuples
        def impl(filter_value):  # pragma: no cover
            content_str = ", ".join(
                [_get_snowflake_sql_literal_scalar(x) for x in filter_value]
            )
            return f"({content_str})"

        return impl
    elif isinstance(filter_type, scalar_isinstance) or filter_type in scalar_equals:
        return lambda filter_value: _get_snowflake_sql_literal_scalar(
            filter_value
        )  # pragma: no cover
    else:
        raise BodoError(
            f"pd.read_sql(): Internal error, unsupported type {filter_type} used in filter pushdown."
        )
    # TODO: Support more types (i.e. Interval, datetime64, datetime.datetime)


def sql_remove_dead_column(sql_node, column_live_map, equiv_vars, typemap):
    """
    Function that tracks which columns to prune from the SQL node.
    This updates type_usecol_offset which stores which arrays in the
    types will need to actually be loaded.

    This is mapped to the used column names during distributed pass.
    """
    return bodo.ir.connector.base_connector_remove_dead_columns(
        sql_node,
        column_live_map,
        equiv_vars,
        typemap,
        "SQLReader",
        # df_colnames is set to an empty list if the table is dead
        # see 'remove_dead_sql'
        sql_node.df_colnames,
    )


numba.parfors.array_analysis.array_analysis_extensions[
    SqlReader
] = bodo.ir.connector.connector_array_analysis
distributed_analysis.distributed_analysis_extensions[
    SqlReader
] = bodo.ir.connector.connector_distributed_analysis
typeinfer.typeinfer_extensions[SqlReader] = bodo.ir.connector.connector_typeinfer
ir_utils.visit_vars_extensions[SqlReader] = bodo.ir.connector.visit_vars_connector
ir_utils.remove_dead_extensions[SqlReader] = remove_dead_sql
numba.core.analysis.ir_extension_usedefs[
    SqlReader
] = bodo.ir.connector.connector_usedefs
ir_utils.copy_propagate_extensions[SqlReader] = bodo.ir.connector.get_copies_connector
ir_utils.apply_copy_propagate_extensions[
    SqlReader
] = bodo.ir.connector.apply_copies_connector
ir_utils.build_defs_extensions[
    SqlReader
] = bodo.ir.connector.build_connector_definitions
distributed_pass.distributed_run_extensions[SqlReader] = sql_distributed_run
remove_dead_column_extensions[SqlReader] = sql_remove_dead_column
ir_extension_table_column_use[SqlReader] = bodo.ir.connector.connector_table_column_use

# XXX: temporary fix pending Numba's #3378
# keep the compiled functions around to make sure GC doesn't delete them and
# the reference to the dynamic function inside them
# (numba/lowering.py:self.context.add_dynamic_addr ...)
compiled_funcs = []


@numba.njit
def sqlalchemy_check():  # pragma: no cover
    with numba.objmode():
        sqlalchemy_check_()


def sqlalchemy_check_():  # pragma: no cover
    try:
        import sqlalchemy  # noqa
    except ImportError:
        message = (
            "Using URI string without sqlalchemy installed."
            " sqlalchemy can be installed by calling"
            " 'conda install -c conda-forge sqlalchemy'."
        )
        raise BodoError(message)


@numba.njit
def pymysql_check():
    """MySQL Check that user has pymysql installed."""
    with numba.objmode():
        pymysql_check_()


def pymysql_check_():  # pragma: no cover
    try:
        import pymysql  # noqa
    except ImportError:
        message = (
            "Using MySQL URI string requires pymsql to be installed."
            " It can be installed by calling"
            " 'conda install -c conda-forge pymysql'"
            " or 'pip install PyMySQL'."
        )
        raise BodoError(message)


@numba.njit
def cx_oracle_check():
    """Oracle Check that user has cx_oracle installed."""
    with numba.objmode():
        cx_oracle_check_()


def cx_oracle_check_():  # pragma: no cover
    try:
        import cx_Oracle  # noqa
    except ImportError:
        message = (
            "Using Oracle URI string requires cx_oracle to be installed."
            " It can be installed by calling"
            " 'conda install -c conda-forge cx_oracle'"
            " or 'pip install cx-Oracle'."
        )
        raise BodoError(message)


@numba.njit
def psycopg2_check():  # pragma: no cover
    """PostgreSQL Check that user has psycopg2 installed."""
    with numba.objmode():
        psycopg2_check_()


def psycopg2_check_():  # pragma: no cover
    try:
        import psycopg2  # noqa
    except ImportError:
        message = (
            "Using PostgreSQL URI string requires psycopg2 to be installed."
            " It can be installed by calling"
            " 'conda install -c conda-forge psycopg2'"
            " or 'pip install psycopg2'."
        )
        raise BodoError(message)


def req_limit(sql_request):
    """
    Processes a SQL requests and search for a LIMIT in the outermost
    query. If it encounters just a limit, it returns the max rows requested.
    Otherwise, it returns None (which incidates a count calculation will need
    to be added to the query).
    """
    import re

    # Regex checks that a query ends with "LIMIT NUM_ROWS"
    # ignoring any surrounding whitespace
    #
    # This should always refer to the outermost table
    # (because inner tables should always be wrapped in parentheses).
    # This regex may fail to detect the limit in some cases, but it
    # shouldn't ever incorrectly detect a limit.
    #
    # TODO: Replace a proper SQL parser (i.e. BodoSQL).
    limit_regex = re.compile(r"LIMIT\s+(\d+)\s*$", re.IGNORECASE)
    m = limit_regex.search(sql_request)
    if m:
        return int(m.group(1))
    else:
        return None


def _gen_sql_reader_py(
    col_names,
    col_typs,
    index_column_name,
    index_column_type,
    type_usecol_offset,
    typingctx,
    targetctx,
    db_type,
    limit,
    parallel,
    is_select_query,
):
    # a unique int used to create global variables with unique names
    call_id = next_label()

    # Prune the columns to only those that are used.
    used_col_names = [col_names[i] for i in type_usecol_offset]
    used_col_types = [col_typs[i] for i in type_usecol_offset]
    if index_column_name:
        used_col_names.append(index_column_name)
        used_col_types.append(index_column_type)

    # See old method in Confluence (Search "multiple_access_by_block")
    # This is a more advanced downloading procedure. It downloads data in an
    # ordered way.
    #
    # Algorithm:
    # ---First determine the number of rows by encapsulating the sql_request
    #    into another one.
    # ---Then broadcast the value obtained to other nodes.
    # ---Then each MPI node downloads the data that he is interested in.
    #    (This is achieved by putting parenthesis under the original SQL request)
    # By doing so we guarantee that the partition is ordered and this guarantees
    # coherency.
    #
    # POSSIBLE IMPROVEMENTS:
    #
    # Sought algorithm: Have a C++ program doing the downloading by blocks and dispatch
    # to other nodes. If ordered is required then do a needed shuffle.
    #
    # For the type determination: If compilation cannot be done in parallel then
    # maybe create a process that access the table type and store them for further
    # usage.
    table_idx = None
    type_usecols_offsets_arr = None
    py_table_type = types.none
    if type_usecol_offset:
        # Create the table type.
        py_table_type = TableType(tuple(col_typs))
    func_text = "def sql_reader_py(sql_request, conn):\n"
    if db_type == "snowflake":
        func_text += f"  ev = bodo.utils.tracing.Event('read_snowflake', {parallel})\n"

        def is_nullable(typ):  # TODO refactor
            return (
                bodo.utils.utils.is_array_typ(typ, False)
                and (
                    not isinstance(typ, types.Array)  # or is_dtype_nullable(typ.dtype)
                )
                and not isinstance(typ, bodo.DatetimeArrayType)
            )

        nullable_cols = [int(is_nullable(col_typs[i])) for i in type_usecol_offset]
        # Handle if we need to append an index
        if index_column_name:
            nullable_cols.append(int(is_nullable(index_column_type)))
        func_text += f"  out_table = snowflake_read(unicode_to_utf8(sql_request), unicode_to_utf8(conn), {parallel}, {len(nullable_cols)}, np.array({nullable_cols}, dtype=np.int32).ctypes)\n"
        func_text += "  check_and_propagate_cpp_exception()\n"
        if index_column_name:
            # The index is always placed in the last slot of the query if it exists.
            func_text += f"  index_var = info_to_array(info_from_table(out_table, {len(type_usecol_offset)}), index_col_typ)\n"
        else:
            # There is no index to load
            func_text += "  index_var = None\n"
        if type_usecol_offset:
            # Map each logical column in the table to its location
            # in the input SQL table
            idx = []
            j = 0
            for i in range(len(col_names)):
                if j < len(type_usecol_offset) and i == type_usecol_offset[j]:
                    idx.append(j)
                    j += 1
                else:
                    idx.append(-1)
            table_idx = np.array(idx, dtype=np.int64)
            func_text += f"  table_var = cpp_table_to_py_table(out_table, table_idx_{call_id}, py_table_type_{call_id})\n"
        else:
            # We only load the index as the table is dead.
            func_text += "  table_var = None\n"
        func_text += "  delete_table(out_table)\n"
        func_text += f"  ev.finalize()\n"
    else:
        if type_usecol_offset:
            # Indicate which columns to load from the table
            func_text += f"  type_usecols_offsets_arr_{call_id}_2 = type_usecols_offsets_arr_{call_id}\n"
            type_usecols_offsets_arr = np.array(type_usecol_offset, dtype=np.int64)
        func_text += "  df_typeref_2 = df_typeref\n"
        func_text += "  sqlalchemy_check()\n"
        if db_type == "mysql" or db_type == "mysql+pymysql":
            func_text += "  pymysql_check()\n"
        elif db_type == "oracle":
            func_text += "  cx_oracle_check()\n"
        elif db_type == "postgresql" or db_type == "postgresql+psycopg2":
            func_text += "  psycopg2_check()\n"

        if parallel and is_select_query:
            # NOTE: assigning a new variable to make globals used inside objmode local to the
            # function, which avoids objmode caching errors
            func_text += "  rank = bodo.libs.distributed_api.get_rank()\n"
            if limit is not None:
                func_text += f"  nb_row = {limit}\n"
            else:
                func_text += '  with objmode(nb_row="int64"):\n'
                func_text += f"     if rank == {MPI_ROOT}:\n"
                func_text += "         sql_cons = 'select count(*) from (' + sql_request + ') x'\n"
                func_text += "         frame = pd.read_sql(sql_cons, conn)\n"
                func_text += "         nb_row = frame.iat[0,0]\n"
                func_text += "     else:\n"
                func_text += "         nb_row = 0\n"
                func_text += "  nb_row = bcast_scalar(nb_row)\n"
            func_text += f"  with objmode(table_var=py_table_type_{call_id}, index_var=index_col_typ):\n"
            func_text += "    offset, limit = bodo.libs.distributed_api.get_start_count(nb_row)\n"
            # https://docs.oracle.com/javadb/10.8.3.0/ref/rrefsqljoffsetfetch.html
            if db_type == "oracle":
                func_text += f"    sql_cons = 'select * from (' + sql_request + ') OFFSET ' + str(offset) + ' ROWS FETCH NEXT ' + str(limit) + ' ROWS ONLY'\n"
            else:
                func_text += f"    sql_cons = 'select * from (' + sql_request + ') x LIMIT ' + str(limit) + ' OFFSET ' + str(offset)\n"

            func_text += "    df_ret = pd.read_sql(sql_cons, conn)\n"
            func_text += (
                "    bodo.ir.connector.cast_float_to_nullable(df_ret, df_typeref_2)\n"
            )
        else:
            func_text += f"  with objmode(table_var=py_table_type_{call_id}, index_var=index_col_typ):\n"
            func_text += "    df_ret = pd.read_sql(sql_request, conn)\n"
            func_text += (
                "    bodo.ir.connector.cast_float_to_nullable(df_ret, df_typeref_2)\n"
            )
        if index_column_name:
            func_text += (
                f"    index_var = df_ret.iloc[:, {len(type_usecol_offset)}].values\n"
            )
            func_text += f"    df_ret.drop(columns=df_ret.columns[{len(type_usecol_offset)}], inplace=True)\n"
        else:
            # Dead Index
            func_text += "    index_var = None\n"
        if type_usecol_offset:
            func_text += f"    arrs = []\n"
            func_text += f"    for i in range(df_ret.shape[1]):\n"
            func_text += f"      arrs.append(df_ret.iloc[:, i].values)\n"
            # Bodo preserves all of the original types needed at typing in col_typs
            func_text += f"    table_var = Table(arrs, type_usecols_offsets_arr_{call_id}_2, {len(col_names)})\n"
        else:
            # Dead Table
            func_text += "    table_var = None\n"
    func_text += "  return (table_var, index_var)\n"

    glbls = globals()  # TODO: fix globals after Numba's #3355 is resolved
    glbls.update(
        {
            "bodo": bodo,
            f"py_table_type_{call_id}": py_table_type,
            "index_col_typ": index_column_type,
        }
    )
    if db_type == "snowflake":
        glbls.update(
            {
                "np": np,
                "unicode_to_utf8": unicode_to_utf8,
                "check_and_propagate_cpp_exception": check_and_propagate_cpp_exception,
                "snowflake_read": _snowflake_read,
                "info_to_array": info_to_array,
                "info_from_table": info_from_table,
                "delete_table": delete_table,
                "cpp_table_to_py_table": cpp_table_to_py_table,
                f"table_idx_{call_id}": table_idx,
            }
        )
    else:
        glbls.update(
            {
                "sqlalchemy_check": sqlalchemy_check,
                "pd": pd,
                "objmode": objmode,
                "bcast_scalar": bcast_scalar,
                "pymysql_check": pymysql_check,
                "cx_oracle_check": cx_oracle_check,
                "psycopg2_check": psycopg2_check,
                "df_typeref": bodo.DataFrameType(
                    tuple(used_col_types),
                    bodo.RangeIndexType(None),
                    tuple(used_col_names),
                ),
                "Table": Table,
                f"type_usecols_offsets_arr_{call_id}": type_usecols_offsets_arr,
            }
        )

    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    sql_reader_py = loc_vars["sql_reader_py"]

    # TODO: no_cpython_wrapper=True crashes for some reason
    jit_func = numba.njit(sql_reader_py)
    compiled_funcs.append(jit_func)
    return jit_func


_snowflake_read = types.ExternalFunction(
    "snowflake_read",
    table_type(
        types.voidptr,
        types.voidptr,
        types.boolean,
        types.int64,
        types.voidptr,
    ),
)

import llvmlite.binding as ll

from bodo.io import arrow_cpp

ll.add_symbol("snowflake_read", arrow_cpp.snowflake_read)
