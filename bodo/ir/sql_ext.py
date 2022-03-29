# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Implementation of pd.read_sql in BODO.
We piggyback on the pandas implementation. Future plan is to have a faster
version for this task.
"""

import numba
import numpy as np
import pandas as pd  # noqa
from numba.core import ir, ir_utils, typeinfer, types
from numba.core.ir_utils import compile_to_numba_ir, replace_arg_nodes

import bodo
import bodo.ir.connector
from bodo import objmode  # noqa
from bodo.ir.csv_ext import _get_dtype_str
from bodo.libs.array import (
    delete_table,
    info_from_table,
    info_to_array,
    table_type,
)
from bodo.libs.distributed_api import bcast, bcast_scalar
from bodo.libs.str_ext import string_type, unicode_to_utf8
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.utils.typing import BodoError
from bodo.utils.utils import check_and_propagate_cpp_exception, sanitize_varname

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
    ):
        self.connector_typ = "sql"
        self.sql_request = sql_request
        self.connection = connection
        self.df_out = df_out  # used only for printing
        self.df_colnames = df_colnames
        # Keep a copy of the column names before running DCE.
        # This is used to detect if we still have unsupported_columns.
        self.original_df_colnames = df_colnames
        self.out_vars = out_vars
        self.out_types = out_types
        # Any columns that had their output name converted by the actual
        # DB result. This is used by Snowflake because we update the SQL query
        # to perform dce and we must specify the exact column name (because we quote
        # escape the names).
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

    def __repr__(self):  # pragma: no cover
        return "{} = ReadSql(sql_request={}, connection={}, col_names={}, original_col_names={}, types={}, vars={}, limit={}, unsupported_columns={}, unsupported_arrow_types={})".format(
            self.df_out,
            self.sql_request,
            self.connection,
            self.df_colnames,
            self.original_df_colnames,
            self.out_types,
            self.out_vars,
            self.limit,
            self.unsupported_columns,
            self.unsupported_arrow_types,
        )


def remove_dead_sql(
    sql_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):
    # TODO
    new_df_colnames = []
    new_out_vars = []
    new_out_types = []

    for i, col_var in enumerate(sql_node.out_vars):
        if col_var.name in lives:
            new_df_colnames.append(sql_node.df_colnames[i])
            new_out_vars.append(sql_node.out_vars[i])
            new_out_types.append(sql_node.out_types[i])

    sql_node.df_colnames = new_df_colnames
    sql_node.out_vars = new_out_vars
    sql_node.out_types = new_out_types

    if len(sql_node.out_vars) == 0:
        return None

    return sql_node


def sql_distributed_run(
    sql_node, array_dists, typemap, calltypes, typingctx, targetctx
):
    # Add debug info about column pruning
    if bodo.user_logging.get_verbose_level() >= 1:
        pruning_msg = "Finish column pruning on read_sql node:\n%s\nColumns loaded %s\n"
        sql_source = sql_node.loc.strformat()
        sql_cols = sql_node.df_colnames
        bodo.user_logging.log_message(
            "Column Pruning",
            pruning_msg,
            sql_source,
            sql_cols,
        )
        # Log if any columns use dictionary encoded arrays.
        dict_encoded_cols = [
            c
            for i, c in enumerate(sql_node.df_colnames)
            if isinstance(
                sql_node.out_types[i], bodo.libs.dict_arr_ext.DictionaryArrayType
            )
        ]
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

    parallel = False
    if array_dists is not None:
        parallel = True
        for v in sql_node.out_vars:
            if (
                array_dists[v.name] != distributed_pass.Distribution.OneD
                and array_dists[v.name] != distributed_pass.Distribution.OneD_Var
            ):
                parallel = False

    # Check for any unsupported columns still remaining
    if sql_node.unsupported_columns:
        # Determine the columns that were eliminated.
        unsupported_cols_set = set(sql_node.unsupported_columns)
        # Compute the intersection of what was kept.
        remaining_unsupported = set()
        name_idx = 0
        for name in sql_node.df_colnames:
            while sql_node.original_df_colnames[name_idx] != name:
                name_idx += 1
            if name_idx in unsupported_cols_set:
                remaining_unsupported.add(name_idx)
            name_idx += 1

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

    n_cols = len(sql_node.out_vars)
    arg_names = ", ".join("arr" + str(i) for i in range(n_cols))
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
    func_text += "    ({},) = _sql_reader_py(sql_request, conn)\n".format(arg_names)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    sql_impl = loc_vars["sql_impl"]

    sql_reader_py = _gen_sql_reader_py(
        sql_node.df_colnames,
        sql_node.out_types,
        typingctx,
        targetctx,
        sql_node.db_type,
        sql_node.limit,
        sql_node.converted_colnames,
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

        # Update the SQL request to remove any unused columns. This is both
        # an optimization (the SQL engine loads less data) and is needed for
        # correctness. See test_sql_snowflake_single_column
        col_str = escape_column_names(
            sql_node.df_colnames, sql_node.db_type, sql_node.converted_colnames
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
    for i in range(len(sql_node.out_vars)):
        nodes[-len(sql_node.out_vars) + i].target = sql_node.out_vars[i]

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
        used_colnames = [x.upper() if x in converted_colnames else x for x in col_names]
        col_str = ", ".join([f'"{x}"' for x in used_colnames])

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


# XXX: temporary fix pending Numba's #3378
# keep the compiled functions around to make sure GC doesn't delete them and
# the reference to the dynamic function inside them
# (numba/lowering.py:self.context.add_dynamic_addr ...)
compiled_funcs = []


@numba.njit
def sqlalchemy_check():
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
        import cx_oracle  # noqa
    except ImportError:
        message = (
            "Using Oracle URI string requires cx_oracle to be installed."
            " It can be installed by calling"
            " 'conda install -c conda-forge cx_oracle'"
            " or 'pip install cx-Oracle'."
        )
        raise BodoError(message)


@numba.njit
def psycopg2_check():
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
    typingctx,
    targetctx,
    db_type,
    limit,
    converted_colnames,
    parallel,
    is_select_query,
):
    sanitized_cnames = [sanitize_varname(c) for c in col_names]
    typ_strs = [
        "{}='{}'".format(s_cname, _get_dtype_str(t))
        for s_cname, t in zip(sanitized_cnames, col_typs)
    ]
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
    if bodo.sql_access_method == "multiple_access_nb_row_first":
        func_text = "def sql_reader_py(sql_request, conn):\n"
        if db_type == "snowflake":
            local_types = {}
            for i, c_typ in enumerate(col_typs):
                local_types[f"col_{i}_type"] = c_typ
            func_text += (
                f"  ev = bodo.utils.tracing.Event('read_snowflake', {parallel})\n"
            )

            def is_nullable(typ):  # TODO refactor
                return bodo.utils.utils.is_array_typ(typ, False) and (
                    not isinstance(typ, types.Array)  # or is_dtype_nullable(typ.dtype)
                )

            nullable_cols = [int(is_nullable(c_typ)) for c_typ in col_typs]
            func_text += f"  out_table = snowflake_read(unicode_to_utf8(sql_request), unicode_to_utf8(conn), {parallel}, {len(col_names)}, np.array({nullable_cols}, dtype=np.int32).ctypes)\n"
            func_text += "  check_and_propagate_cpp_exception()\n"
            for i, c_name in enumerate(sanitized_cnames):
                func_text += f"  {c_name} = info_to_array(info_from_table(out_table, {i}), col_{i}_type)\n"
            func_text += "  delete_table(out_table)\n"
            func_text += f"  ev.finalize()\n"
        else:
            func_text += "  sqlalchemy_check()\n"
            if db_type == "mysql" or db_type == "mysql+pymysql":
                func_text += "  pymysql_check()\n"
            elif db_type == "oracle":
                func_text += "  cx_oracle_check()\n"
            elif db_type == "postgresql" or db_type == "postgresql+psycopg2":
                func_text += "  psycopg2_check()\n"

            if parallel and is_select_query:
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
                func_text += "  with objmode({}):\n".format(", ".join(typ_strs))
                func_text += "    offset, limit = bodo.libs.distributed_api.get_start_count(nb_row)\n"
                # Update the SQL request to remove any unused columns. This is both
                # an optimization (the SQL engine loads less data) and is needed for
                # correctness. See test_read_sql_column_function
                col_str = escape_column_names(col_names, db_type, converted_colnames)
                # https://docs.oracle.com/javadb/10.8.3.0/ref/rrefsqljoffsetfetch.html
                if db_type == "oracle":
                    func_text += f"    sql_cons = 'select {col_str} from (' + sql_request + ') OFFSET ' + str(offset) + ' ROWS FETCH NEXT ' + str(limit) + ' ROWS ONLY'\n"
                else:
                    func_text += f"    sql_cons = 'select {col_str} from (' + sql_request + ') x LIMIT ' + str(limit) + ' OFFSET ' + str(offset)\n"

                func_text += "    df_ret = pd.read_sql(sql_cons, conn)\n"
            else:
                func_text += "  with objmode({}):\n".format(", ".join(typ_strs))
                func_text += "    df_ret = pd.read_sql(sql_request, conn)\n"
            # We assumed that sanitized_cnames and col_names are list of strings
            for s_cname, cname in zip(sanitized_cnames, col_names):
                func_text += "    {} = df_ret['{}'].values\n".format(s_cname, cname)
        func_text += "  return ({},)\n".format(", ".join(sc for sc in sanitized_cnames))

    glbls = {"bodo": bodo}
    if db_type == "snowflake":
        glbls.update(local_types)
        glbls.update(
            {
                "np": np,
                "unicode_to_utf8": unicode_to_utf8,
                "check_and_propagate_cpp_exception": check_and_propagate_cpp_exception,
                "snowflake_read": _snowflake_read,
                "info_to_array": info_to_array,
                "info_from_table": info_from_table,
                "delete_table": delete_table,
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
