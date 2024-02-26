# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implementation of pd.read_sql in Bodo.
We piggyback on the pandas implementation. Future plan is to have a faster
version for this task.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    NamedTuple,
    Optional,
    Union,
)
from urllib.parse import urlparse

import llvmlite.binding as ll
import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
from llvmlite import ir as lir
from numba.core import cgutils, ir, ir_utils, typeinfer, types
from numba.core.ir_utils import (
    compile_to_numba_ir,
    next_label,
    replace_arg_nodes,
)
from numba.extending import intrinsic

import bodo
import bodo.ir.connector
import bodo.user_logging
from bodo import objmode
from bodo.hiframes.table import Table, TableType
from bodo.io import arrow_cpp
from bodo.io.arrow_reader import ArrowReaderType
from bodo.io.helpers import map_cpp_to_py_table_column_idxs, pyarrow_schema_type
from bodo.io.parquet_pio import ParquetPredicateType
from bodo.ir.connector import Connector
from bodo.ir.filter import Filter, supported_funcs_map
from bodo.libs.array import (
    array_from_cpp_table,
    cpp_table_to_py_table,
    delete_table,
    table_type,
)
from bodo.libs.dict_arr_ext import dict_str_arr_type
from bodo.libs.distributed_api import bcast, bcast_scalar
from bodo.libs.str_ext import string_type, unicode_to_utf8
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.transforms.distributed_analysis import Distribution
from bodo.transforms.table_column_del_pass import (
    ir_extension_table_column_use,
    remove_dead_column_extensions,
)
from bodo.utils.typing import (
    BodoError,
    get_overload_const_str,
    is_nullable_ignore_sentinals,
    is_overload_constant_str,
)
from bodo.utils.utils import (
    check_and_propagate_cpp_exception,
    inlined_check_and_propagate_cpp_exception,
)

if TYPE_CHECKING:  # pragma: no cover
    from llvmlite.ir.builder import IRBuilder
    from numba.core.base import BaseContext


ll.add_symbol("snowflake_read_py_entry", arrow_cpp.snowflake_read_py_entry)
ll.add_symbol(
    "snowflake_reader_init_py_entry", arrow_cpp.snowflake_reader_init_py_entry
)
ll.add_symbol(
    "snowflake_reader_init_py_entry", arrow_cpp.snowflake_reader_init_py_entry
)
ll.add_symbol("arrow_reader_read_py_entry", arrow_cpp.arrow_reader_read_py_entry)
ll.add_symbol("iceberg_pq_read_py_entry", arrow_cpp.iceberg_pq_read_py_entry)
ll.add_symbol(
    "iceberg_pq_reader_init_py_entry", arrow_cpp.iceberg_pq_reader_init_py_entry
)

MPI_ROOT = 0


class SnowflakeReadParams(NamedTuple):
    """Common inputs into snowflake reader functions."""

    snowflake_dict_cols_array: npt.NDArray[np.int32]
    nullable_cols_array: npt.NDArray[np.int32]

    @classmethod
    def from_column_information(
        cls,
        out_used_cols: list[int],
        col_typs: list[types.ArrayCompatible],
        index_column_name: Optional[str],
        index_column_type: Union[types.ArrayCompatible, types.NoneType],
    ):  # pragma: no cover
        """Construct a SnowflakeReaderParams from the IR parameters"""
        col_indices_map = {c: i for i, c in enumerate(out_used_cols)}
        snowflake_dict_cols = [
            col_indices_map[i]
            for i in out_used_cols
            if col_typs[i] == dict_str_arr_type
        ]

        nullable_cols = [
            int(is_nullable_ignore_sentinals(col_typs[i])) for i in out_used_cols
        ]
        # Handle if we need to append an index
        if index_column_name:
            nullable_cols.append(int(is_nullable_ignore_sentinals(index_column_type)))
        snowflake_dict_cols_array = np.array(snowflake_dict_cols, dtype=np.int32)
        nullable_cols_array = np.array(nullable_cols, dtype=np.int32)

        return cls(
            snowflake_dict_cols_array=snowflake_dict_cols_array,
            nullable_cols_array=nullable_cols_array,
        )


class SqlReader(Connector):
    connector_typ: str = "sql"

    def __init__(
        self,
        sql_request: str,
        connection: str,
        df_out_varname: str,
        out_table_col_names: list[str],
        out_table_col_types: list[types.ArrayCompatible],
        out_vars: list[ir.Var],
        converted_colnames: list[str],
        db_type: str,
        loc: ir.Loc,
        unsupported_columns: list[str],
        unsupported_arrow_types: list[pa.DataType],
        is_select_query: bool,
        has_side_effects: bool,
        index_column_name: Optional[str],
        index_column_type: Union[types.ArrayCompatible, types.NoneType],
        database_schema: Optional[str],
        # Only relevant for Iceberg and Snowflake
        pyarrow_schema: Optional[pa.Schema],
        # Only relevant for Iceberg MERGE INTO COW
        is_merge_into: bool,
        file_list_type: types.Type,
        snapshot_id_type: types.Type,
        # Runtime should downcast decimal columns to double
        # Only relevant for Snowflake ATM
        downcast_decimal_to_double: bool,
        # Batch size to read chunks in, or none, to read the entire table together
        # Only supported for Snowflake
        # Treated as compile-time constant for simplicity
        # But not enforced that all chunks are this size
        chunksize: Optional[int] = None,
    ):
        # Column Names and Types. Common for all Connectors
        # - Output Columns
        # - Original Columns
        # - Index Column
        # - Unsupported Columns
        self.out_table_col_names = out_table_col_names
        self.out_table_col_types = out_table_col_types
        # Both are None if index=False
        self.index_column_name = index_column_name
        self.index_column_type = index_column_type
        # These fields are used to enable compilation if unsupported columns
        # get eliminated. Currently only used with snowflake.
        self.unsupported_columns = unsupported_columns
        self.unsupported_arrow_types = unsupported_arrow_types

        self.sql_request = sql_request
        self.connection = connection
        self.df_out_varname = df_out_varname  # used only for printing
        self.out_vars = out_vars
        # Any columns that had their output name converted by the actual
        # DB result. This is used by Snowflake because we update the SQL query
        # to perform dce and we must specify the exact column name (because we quote
        # escape the names). This may include both the table column names and the
        # index column.
        self.converted_colnames = converted_colnames
        self.loc = loc
        self.limit = req_limit(sql_request)
        self.db_type = db_type
        # Support for filter pushdown. Currently only used with snowflake
        # and iceberg.
        self.filters = None

        self.is_select_query = is_select_query
        # Does this query have side effects (e.g. DELETE). If so
        # we cannot perform DCE on the whole node.
        self.has_side_effects = has_side_effects

        # List of indices within the table name that are used.
        # out_table_col_names is unchanged unless the table is deleted,
        # so this is used to track dead columns.
        self.out_used_cols = list(range(len(out_table_col_names)))
        # The database schema used to load data. This is currently only
        # supported/required for snowflake and must be provided
        # at compile time.
        self.database_schema = database_schema
        # This is the PyArrow schema object.
        # Only relevant for Iceberg at the moment,
        # but potentially for Snowflake in the future
        self.pyarrow_schema = pyarrow_schema
        # Is this table load done as part of a merge into operation.
        # If so we have special behavior regarding filtering.
        self.is_merge_into = is_merge_into
        # Is the variable currently alive. This should be replaced with more
        # robust handling in connectors.
        self.is_live_table = True
        # Set if we are loading the file list and snapshot_id for iceberg.
        self.file_list_live = is_merge_into
        self.snapshot_id_live = is_merge_into
        if is_merge_into:
            self.file_list_type = file_list_type
            self.snapshot_id_type = snapshot_id_type
        else:
            self.file_list_type = types.none
            self.snapshot_id_type = types.none

        self.downcast_decimal_to_double = downcast_decimal_to_double
        self.chunksize = chunksize

    def __repr__(self) -> str:  # pragma: no cover
        out_varnames = tuple(v.name for v in self.out_vars)
        return f"{out_varnames} = SQLReader(sql_request={self.sql_request}, connection={self.connection}, out_col_names={self.out_table_col_names}, out_col_types={self.out_table_col_types}, df_out_varname={self.df_out_varname}, limit={self.limit}, unsupported_columns={self.unsupported_columns}, unsupported_arrow_types={self.unsupported_arrow_types}, is_select_query={self.is_select_query}, index_column_name={self.index_column_name}, index_column_type={self.index_column_type}, out_used_cols={self.out_used_cols}, database_schema={self.database_schema}, pyarrow_schema={self.pyarrow_schema}, is_merge_into={self.is_merge_into}, downcast_decimal_to_double={self.downcast_decimal_to_double})"

    def out_vars_and_types(self) -> list[tuple[str, types.Type]]:
        if self.is_streaming:
            return [
                (
                    self.out_vars[0].name,
                    ArrowReaderType(self.out_table_col_names, self.out_table_col_types),
                )
            ]
        vars = [
            (self.out_vars[0].name, TableType(tuple(self.out_table_col_types))),
            (self.out_vars[1].name, self.index_column_type),
        ]
        if len(self.out_vars) > 2:
            vars.append((self.out_vars[2].name, self.file_list_type))
        if len(self.out_vars) > 3:
            vars.append((self.out_vars[3].name, self.snapshot_id_type))
        return vars

    def out_table_distribution(self) -> Distribution:
        if not self.is_select_query:
            return Distribution.REP
        elif self.limit is not None:
            return Distribution.OneD_Var
        else:
            return Distribution.OneD


def parse_dbtype(con_str) -> tuple[str, str]:
    """
    Converts a constant string used for db_type to a standard representation
    for each database.
    """
    parseresult = urlparse(con_str)
    db_type = parseresult.scheme
    con_paswd = parseresult.password
    # urlparse skips oracle since its handle has _
    # which is not in `scheme_chars`
    # oracle+cx_oracle
    if con_str.startswith("oracle+cx_oracle://"):
        return "oracle", con_paswd
    if db_type == "mysql+pymysql":
        # Standardize mysql to always use "mysql"
        return "mysql", con_paswd

    # NOTE: if you're updating supported schemes here, don't forget
    # to update the associated error message in _run_call_read_sql_table

    if con_str.startswith("iceberg+glue") or parseresult.scheme in (
        "iceberg",
        "iceberg+file",
        "iceberg+s3",
        "iceberg+thrift",
        "iceberg+http",
        "iceberg+https",
    ):
        # Standardize iceberg to always use "iceberg"
        return "iceberg", con_paswd
    return db_type, con_paswd


def remove_iceberg_prefix(con: str) -> str:
    import sys

    # Remove Iceberg Prefix when using Internally
    # For support before Python 3.9
    # TODO: Remove after deprecating Python 3.8
    if sys.version_info.minor < 9:  # pragma: no cover
        if con.startswith("iceberg+"):
            con = con[len("iceberg+") :]
        if con.startswith("iceberg://"):
            con = con[len("iceberg://") :]
    else:
        con = con.removeprefix("iceberg+").removeprefix("iceberg://")
    return con


def remove_dead_sql(
    sql_node: SqlReader,
    lives_no_aliases,
    lives,
    arg_aliases,
    alias_map,
    func_ir,
    typemap,
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
    if sql_node.is_streaming:  # pragma: no cover
        return sql_node

    table_var = sql_node.out_vars[0].name
    index_var = sql_node.out_vars[1].name
    file_list_var = sql_node.out_vars[2].name if len(sql_node.out_vars) > 2 else None
    snapshot_id_var = sql_node.out_vars[3].name if len(sql_node.out_vars) > 3 else None
    if (
        not sql_node.has_side_effects
        and table_var not in lives
        and index_var not in lives
        and file_list_var not in lives
        and snapshot_id_var not in lives
    ):
        # If neither the table or index is live and it has
        # no side effects, remove the node.
        return None

    if table_var not in lives:
        # If table isn't live we mark the out_table_col_names as empty
        # and avoid loading the table
        sql_node.out_table_col_names = []
        sql_node.out_table_col_types = []
        sql_node.out_used_cols = []
        sql_node.is_live_table = False

    if index_var not in lives:
        # If the index_var not in lives we don't load the index.
        # To do this we mark the index_column_name as None
        sql_node.index_column_name = None
        sql_node.index_column_type = types.none

    if file_list_var not in lives:
        sql_node.file_list_live = False
        sql_node.file_list_type = types.none

    if snapshot_id_var not in lives:
        sql_node.snapshot_id_live = False
        sql_node.snapshot_id_type = types.none
    return sql_node


def _get_sql_column_str(
    p0: Union[str, Filter],
    scalars_to_unpack,
    converted_colnames: Iterable[str],
    filter_map: dict[str, str],
    typemap,
) -> str:  # pragma: no cover
    """get SQL code for representing a column in filter pushdown.
    E.g. WHERE  ( ( coalesce(\"L_COMMITDATE\", {f0}) >= {f1} ) )
                    ^^^^^^^^^^^^^^^^^^^^^^^^^

    Args:
        p0: column name or tuple representing the computation for the column
        converted_colnames: column names that need converted to upper case
        filter_map: map of IR variable names to read function variable
            names. E.g. {'_v14call_method_6_224': 'f0'}

    Returns:
        str: code representing the column (e.g. "A", "coalesce(\"L_COMMITDATE\", {f0})")
    """

    if isinstance(p0, str):
        assert isinstance(p0, str), "_get_sql_column_str: expected string column name"
        col_name = convert_col_name(p0, converted_colnames)
        return '\\"' + col_name + '\\"'

    assert isinstance(p0, tuple), "_get_sql_column_str: expected filter tuple"
    col_name = _get_sql_column_str(
        p0[0], scalars_to_unpack, converted_colnames, filter_map, typemap
    )
    kernel_func, func_args = p0[1], p0[2]
    args_name = func_args.name

    if kernel_func not in supported_funcs_map:
        raise NotImplementedError(
            f"Filter pushdown not implemented for {kernel_func} function"
        )

    sql_func = supported_funcs_map[kernel_func]
    read_func_var = filter_map[args_name]
    args_type_var = typemap[args_name]

    args_str = ""

    if isinstance(func_args, ir.Var):
        if isinstance(args_type_var, types.BaseTuple):
            scalars_to_unpack.append(args_name)

            # If it's a tuple object, we need to iterate through
            # the tuple to generate a filter of the form:
            # "coalesce(\"L_COMMITDATE\", {f0[0]}, {f0[1]}) >= {f1}"
            for i in range(len(args_type_var)):
                args_str += f", {{{read_func_var}[{i}]}}"

        # We need to distinguish whether or not this function
        # takes in additional args, via the name of the IR variable
        elif "dummy" not in func_args.name:
            args_str = f", {{{read_func_var}}}"
    else:
        args_str = f", {func_args}"

    return f"{sql_func}({col_name}{args_str})"


def sql_distributed_run(
    sql_node: SqlReader,
    array_dists,
    typemap,
    calltypes,
    typingctx,
    targetctx,
    is_independent: bool = False,
    meta_head_only_info=None,
):
    # Add debug info about column pruning
    if bodo.user_logging.get_verbose_level() >= 1:
        pruning_msg = "Finish column pruning on read_sql node:\n%s\nColumns loaded %s\n"
        sql_cols = []
        sql_types = []
        dict_encoded_cols = []
        out_types = sql_node.out_table_col_types
        for i in sql_node.out_used_cols:
            colname = sql_node.out_table_col_names[i]
            sql_cols.append(colname)
            sql_types.append(out_types[i])
            if isinstance(out_types[i], bodo.libs.dict_arr_ext.DictionaryArrayType):
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
        if bodo.user_logging.get_verbose_level() >= 2:
            io_msg = "read_sql table/query:\n%s\n\nColumns/Types:\n"
            for c, t in zip(sql_cols, sql_types):
                io_msg += f"{c}: {t}\n"
            bodo.user_logging.log_message(
                "SQL I/O",
                io_msg,
                sql_node.sql_request,
            )

    if sql_node.is_streaming:  # pragma: no cover
        parallel = bodo.ir.connector.is_chunked_connector_table_parallel(
            sql_node, array_dists, "SQLReader"
        )
    else:
        parallel = bodo.ir.connector.is_connector_table_parallel(
            sql_node, array_dists, typemap, "SQLReader"
        )

    # Check for any unsupported columns still remaining
    if sql_node.unsupported_columns:
        # Determine the columns that were eliminated.
        unsupported_cols_set = set(sql_node.unsupported_columns)
        used_cols_set = set(sql_node.out_used_cols)
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
                    f"Column '{sql_node.unsupported_columns[col_num]}' with unsupported arrow type {sql_node.unsupported_arrow_types[idx]}"
                )
                idx += 1
            total_msg = "\n".join(msg_list)
            raise BodoError(total_msg, loc=sql_node.loc)

    # Generate the limit
    if sql_node.limit is None and (
        not meta_head_only_info or meta_head_only_info[0] is None
    ):
        # There is no limit
        limit = None
    elif sql_node.limit is None:
        # There is only limit pushdown
        limit = meta_head_only_info[0]
    elif not meta_head_only_info or meta_head_only_info[0] is None:
        # There is only a limit already in the query
        limit = sql_node.limit
    else:
        # There is limit pushdown and a limit already in the query.
        # Compute the min to minimize compute.
        limit = min(limit, meta_head_only_info[0])

    filter_map, filter_vars = bodo.ir.connector.generate_filter_map(sql_node.filters)
    extra_args = ", ".join(filter_map.values())
    func_text = f"def sql_impl(sql_request, conn, database_schema, {extra_args}):\n"
    # If we are doing regular SQL, filters are embedded into the query.
    # Iceberg passes these to parquet instead.
    if sql_node.is_select_query and sql_node.db_type != "iceberg":
        if sql_node.filters:  # pragma: no cover
            # This path is only taken on Azure because snowflake
            # is not tested on AWS.

            # If a predicate should be and together, they will be multiple tuples within the same list.
            # If predicates should be or together, they will be within separate lists.
            # i.e.
            # [[('l_linestatus', '<>', var1), ('l_shipmode', '=', var2))]]
            # -> -> (l_linestatus <> var1) AND (l_shipmode = var2)
            # [[('l_linestatus', '<>', var1)], [('l_shipmode', '=', var2))]]
            # -> (l_linestatus <> var1) OR (l_shipmode = var2)
            # [[('l_linestatus', '<>', var1)], [('l_shipmode', '=', var2))]]
            or_conds = []
            # Certain scalar values (e.g. like, ilike) are tuples of multiple variables
            # used in the same operation. If they are in this list, rather than convert
            # the tuple to a Snowflake usable literal we convert the individual elements
            # in the tuple to Snowflake usable literals.
            scalars_to_unpack = []
            for and_list in sql_node.filters:
                and_conds = []
                for p in and_list:
                    single_filter = _generate_column_filter(
                        p,
                        scalars_to_unpack,
                        sql_node.converted_colnames,
                        filter_map,
                        typemap,
                    )
                    and_conds.append(" ".join(single_filter))
                or_conds.append(" ( " + " AND ".join(and_conds) + " ) ")
            where_cond = " WHERE " + " OR ".join(or_conds)
            if bodo.user_logging.get_verbose_level() >= 1:
                msg = "SQL filter pushed down:\n%s\n%s\n"
                filter_source = sql_node.loc.strformat()
                bodo.user_logging.log_message(
                    "Filter Pushdown",
                    msg,
                    filter_source,
                    where_cond,
                )
            for ir_varname, arg in filter_map.items():
                if ir_varname in scalars_to_unpack:
                    num_elements = typemap[ir_varname].count
                    elems = ", ".join(
                        [f"get_sql_literal({arg}[{i}])" for i in range(num_elements)]
                    )
                    func_text += f"    {arg} = ({elems},)\n"
                else:
                    func_text += f"    {arg} = get_sql_literal({arg})\n"
            # Append filters via a format string. This format string is created and populated
            # at runtime because filter variables aren't necessarily constants (but they are scalars).
            func_text += f'    sql_request = f"{{sql_request}} {where_cond}"\n'
        # sql_node.limit is the limit value already found in the original sql_request
        # if sql_node.limit == limit then 1 of two things must be True:
        # 1. The limit pushdown value is None. We do not add a limit to the query.
        # 2. meta_head_only_info[0] >= sql_node.limit. If so the limit in the query
        #    is smaller than the limit being pushdown so we can ignore it.
        if sql_node.limit != limit:
            func_text += f'    sql_request = f"{{sql_request}} LIMIT {limit}"\n'

    filter_args = ""
    if sql_node.db_type == "iceberg":
        # Pass args to _sql_reader_py with iceberg
        filter_args = extra_args

    # total_rows is used for setting total size variable below
    if sql_node.is_streaming:  # pragma: no cover
        func_text += f"    snowflake_reader = _sql_reader_py(sql_request, conn, database_schema, {filter_args})\n"
    else:
        func_text += f"    (total_rows, table_var, index_var, file_list, snapshot_id) = _sql_reader_py(sql_request, conn, database_schema, {filter_args})\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    sql_impl = loc_vars["sql_impl"]

    genargs = {
        "col_names": sql_node.out_table_col_names,
        "col_typs": sql_node.out_table_col_types,
        "index_column_name": sql_node.index_column_name,
        "index_column_type": sql_node.index_column_type,
        "out_used_cols": sql_node.out_used_cols,
        "converted_colnames": sql_node.converted_colnames,
        "db_type": sql_node.db_type,
        "limit": limit,
        "parallel": parallel,
        "typemap": typemap,
        "filters": sql_node.filters,
        "pyarrow_schema": sql_node.pyarrow_schema,
        "is_dead_table": not sql_node.is_live_table,
        "is_select_query": sql_node.is_select_query,
        "is_merge_into": sql_node.is_merge_into,
        "is_independent": is_independent,
        "downcast_decimal_to_double": sql_node.downcast_decimal_to_double,
    }
    if sql_node.is_streaming:
        assert sql_node.chunksize is not None
        if sql_node.db_type == "snowflake":  # pragma: no cover
            sql_reader_py = _gen_snowflake_reader_chunked_py(
                **genargs, chunksize=sql_node.chunksize
            )
        else:
            sql_reader_py = _gen_iceberg_reader_chunked_py(
                **genargs,
                chunksize=sql_node.chunksize,
            )
    else:
        sql_reader_py = _gen_sql_reader_py(**genargs)

    schema_type = types.none if sql_node.database_schema is None else string_type
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
        arg_typs=(string_type, string_type, schema_type)
        + tuple(typemap[v.name] for v in filter_vars),
        typemap=typemap,
        calltypes=calltypes,
    ).blocks.popitem()[1]

    if sql_node.is_select_query and sql_node.db_type != "iceberg":
        # Prune the columns to only those that are used.
        # Note: Iceberg skips this step as pruning is done in parquet.
        used_col_names = [
            sql_node.out_table_col_names[i] for i in sql_node.out_used_cols
        ]
        if sql_node.index_column_name:
            used_col_names.append(sql_node.index_column_name)
        if len(used_col_names) == 0:
            # If we are loading 0 columns then replace the query with a COUNT(*)
            # as we just need the length of the table.
            col_str = "COUNT(*)"
        else:
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
            ir.Const(sql_node.database_schema, sql_node.loc),
        ]
        + filter_vars,
    )
    nodes = f_block.body[:-3]

    # Set total size variable if necessary (for limit pushdown, iceberg specific)
    # value comes from 'total_rows' output of '_sql_reader_py' above
    if meta_head_only_info:
        nodes[-5].target = meta_head_only_info[1]

    if sql_node.is_streaming:  # pragma: no cover
        nodes[-1].target = sql_node.out_vars[0]
        return nodes

    # assign output table
    nodes[-4].target = sql_node.out_vars[0]
    # assign output index array
    nodes[-3].target = sql_node.out_vars[1]
    # At most one of the table and the index
    # can be dead because otherwise the whole
    # node should have already been removed.
    assert sql_node.has_side_effects or not (
        sql_node.index_column_name is None and not sql_node.is_live_table
    ), "At most one of table and index should be dead if the SQL IR node is live and has no side effects"
    if sql_node.index_column_name is None:
        # If the index_col is dead, remove the node.
        nodes.pop(-3)
    elif not sql_node.is_live_table:
        # If the table is dead, remove the node
        nodes.pop(-4)

    # Do we load the file_list
    if sql_node.file_list_live:
        nodes[-2].target = sql_node.out_vars[2]
    else:
        nodes.pop(-2)
    # Do we load the snapshot_id
    if sql_node.snapshot_id_live:
        nodes[-1].target = sql_node.out_vars[3]
    else:
        nodes.pop(-1)

    return nodes


def convert_col_name(col_name: str, converted_colnames: Iterable[str]) -> str:
    if col_name in converted_colnames:
        return col_name.upper()
    return col_name


def escape_column_names(col_names, db_type, converted_colnames):
    """
    Function that escapes column names when updating the SQL queries.
    Some outputs (i.e. count(*)) map to both functions and the output
    column names in certain dialects. If these are re-added to the query,
    it may modify the results by rerunning the function, so we must
    escape the column names

    See: test_read_sql_column_function and test_sql_snowflake_count
    """
    # In Snowflake/Oracle we avoid functions by wrapping column names in quotes.
    # This makes the name case sensitive, so we avoid this by undoing any
    # conversions in the output as needed.
    if db_type == "snowflake":
        # Snowflake needs to lower-case names back to uppercase
        # and needs to escape double quotes (by doubling them)
        from bodo.io.snowflake import escape_col_name

        col_str = ", ".join(
            escape_col_name(convert_col_name(x, converted_colnames)) for x in col_names
        )

    elif db_type == "oracle":
        # Oracle needs to convert all lower case strings back to uppercase
        used_col_names = []
        for x in col_names:
            used_col_names.append(convert_col_name(x, converted_colnames))

        col_str = ", ".join([f'"{x}"' for x in used_col_names])

    # MySQL uses tilda as an escape character by default, not quotations
    # However, MySQL does support using quotations in ASCII_MODE. Tilda is always allowed though
    # MySQL names are case-insensitive
    elif db_type == "mysql":
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


def _generate_column_filter(
    filter: Filter,
    scalars_to_unpack,
    converted_colnames: list[str],
    filter_map: dict[str, str],
    typemap: dict[str, types.Type],
) -> list[str]:  # pragma: no cover
    """Generate a filter string

    Args:
        filter (tuple[Union[str, tuple], str, Union[ir.Var, str]]): filter in Bodo format to convert
        scalars_to_unpack (list[str]): A list that will be appended to with any scalars that should be
            unpacked before converting to a Snowflake literal. This is to ensure we have a tuple of literals
            as opposed to a Tuple literal.
        converted_colnames: list of column names that must have their case converted.
        filter_map: Mapping of IR variable name to runtime variable name.
        typemap: Dictionary used to determine scalar types for each variable name.

    Returns:
        The string components to be joined to generate a single filter without AND/OR.
    """
    p0, p1, p2 = filter
    if p1 == "ALWAYS_TRUE":
        # Special operator for True
        return ["(TRUE)"]
    elif p1 == "ALWAYS_FALSE":
        # Special operators for False.
        return ["(FALSE)"]
    elif p1 == "ALWAYS_NULL":
        # Special operators for NULL.
        return ["(NULL)"]
    elif p1 == "not":
        assert not isinstance(p0, str)
        # Not recurses on another function operation.
        inner_filter = _generate_column_filter(
            p0, scalars_to_unpack, converted_colnames, filter_map, typemap
        )
        return ["(", "NOT"] + inner_filter + [")"]
    else:
        # These operators must operate on a column that we will load.
        p0 = _get_sql_column_str(
            p0, scalars_to_unpack, converted_colnames, filter_map, typemap
        )
        # If p2 is a constant that isn't in the IR (i.e. NULL)
        # just load the value directly, otherwise load the variable
        # at runtime.
        scalar_filter = (
            "{" + filter_map[p2.name] + "}" if isinstance(p2, ir.Var) else p2
        )
        if p1 in (
            "startswith",
            "endswith",
            "contains",
        ):
            return ["(", p1, "(", p0, ",", scalar_filter, "))"]
        elif p1 in (
            "case_insensitive_startswith",
            "case_insensitive_endswith",
            "case_insensitive_contains",
            "case_insensitive_equality",
        ):
            op = p1[len("case_insensitive_") :]
            if op == "equality":
                comparison = "="
                # Equality is just =, not a function
                return [
                    "(LOWER(",
                    p0,
                    ")",
                    comparison,
                    "LOWER(",
                    scalar_filter,
                    "))",
                ]
            else:
                return [
                    "(",
                    op,
                    "(LOWER(",
                    p0,
                    "), LOWER(",
                    scalar_filter,
                    ")))",
                ]
        elif p1 in ("like", "ilike"):
            assert isinstance(p2, ir.Var)
            # You can't pass the empty string to escape. As a result we
            # must confirm its not the empty string
            has_escape = True
            escape_typ = typemap[p2.name][1]
            if is_overload_constant_str(escape_typ):
                escape_val = get_overload_const_str(escape_typ)
                has_escape = escape_val != ""
            escape_section = (
                f"escape {{{filter_map[p2.name]}[1]}}" if has_escape else ""
            )
            single_filter = [
                "(",
                p0,
                p1,
                f"{{{filter_map[p2.name]}[0]}} {escape_section}",
                ")",
            ]
            # Indicate the tuple variable is not directly passed to Snowflake, instead its
            # components are.
            scalars_to_unpack.append(p2.name)
            return single_filter
        elif p1 == "REGEXP_LIKE":
            assert isinstance(p2, ir.Var)

            pattern_arg = f"{{{filter_map[p2.name]}[0]}}"
            flags_arg = f"{{{filter_map[p2.name]}[1]}}"

            single_filter = ["(", p1, "(", p0, ",", pattern_arg, ",", flags_arg, "))"]
            # Indicate the tuple variable is not directly passed to Snowflake, instead its
            # components are.
            scalars_to_unpack.append(p2.name)
            return single_filter
        else:
            return ["(", p0, p1, scalar_filter, ")"]


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
        or filter_type == types.bool_
    ):
        # Numeric and boolean values can just return the string representation
        return lambda filter_value: str(filter_value)  # pragma: no cover
    elif isinstance(filter_type, bodo.PandasTimestampType):
        if filter_type.tz is None:
            tz_str = "TIMESTAMP_NTZ"
        else:
            # You cannot specify a specific timestamp so instead we assume
            # we are using the default timezone. This should be fine since the
            # data matches.
            # https://docs.snowflake.com/en/sql-reference/data-types-datetime.html#timestamp-ltz-timestamp-ntz-timestamp-tz
            tz_str = "TIMESTAMP_LTZ"

        # Timestamp needs to be converted to a timestamp literal
        def impl(filter_value):  # pragma: no cover
            nanosecond = filter_value.nanosecond
            nanosecond_prepend = ""
            if nanosecond < 10:
                nanosecond_prepend = "00"
            elif nanosecond < 100:
                nanosecond_prepend = "0"
            # TODO: Refactor once strftime support nanoseconds
            return f"timestamp '{filter_value.strftime('%Y-%m-%d %H:%M:%S.%f')}{nanosecond_prepend}{nanosecond}'::{tz_str}"  # pragma: no cover

        return impl
    elif filter_type == bodo.datetime_date_type:
        # datetime.date needs to be converted to a date literal
        # Just return the string wrapped in quotes.
        # https://docs.snowflake.com/en/sql-reference/data-types-datetime.html#date
        return (
            lambda filter_value: f"date '{filter_value.strftime('%Y-%m-%d')}'"
        )  # pragma: no cover
    elif filter_type == bodo.datetime64ns:
        # datetime64 needs to be a Timestamp literal
        return lambda filter_value: bodo.ir.sql_ext._get_snowflake_sql_literal_scalar(
            pd.Timestamp(filter_value)
        )  # pragma: no cover
    elif filter_type == types.none:
        return lambda filter_value: "NULL"  # pragma: no cover
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
    scalar_isinstance = (types.Integer, types.Float, bodo.PandasTimestampType)
    scalar_equals = (
        bodo.datetime_date_type,
        types.unicode_type,
        types.bool_,
        bodo.datetime64ns,
        types.none,
    )
    filter_type = types.unliteral(filter_value)
    if (
        isinstance(
            filter_type,
            (
                types.List,
                types.Array,
                bodo.IntegerArrayType,
                bodo.FloatingArrayType,
                bodo.DatetimeArrayType,
            ),
        )
        or filter_type
        in (
            bodo.string_array_type,
            bodo.dict_str_arr_type,
            bodo.boolean_array_type,
            bodo.datetime_date_array_type,
        )
    ) and (
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


def sql_remove_dead_column(sql_node: SqlReader, column_live_map, equiv_vars, typemap):
    """
    Function that tracks which columns to prune from the SQL node.
    This updates out_used_cols which stores which arrays in the
    types will need to actually be loaded.

    This is mapped to the used column names during distributed pass.
    """
    return bodo.ir.connector.base_connector_remove_dead_columns(
        sql_node,
        column_live_map,
        equiv_vars,
        typemap,
        "SQLReader",
        # out_table_col_names is set to an empty list if the table is dead
        # see 'remove_dead_sql'
        sql_node.out_table_col_names,
        # Iceberg and Snowflake don't require reading any columns
        require_one_column=sql_node.db_type not in ("iceberg", "snowflake"),
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
    Otherwise, it returns None (which indicates a count calculation will need
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


def prune_columns(
    col_names: list[str],
    col_typs: list[types.ArrayCompatible],
    out_used_cols: list[int],
    index_column_name: Optional[str],
    index_column_type: Union[types.ArrayCompatible, types.NoneType],
):
    """Prune the columns to only those that are used in the snowflake reader."""
    used_col_names = [col_names[i] for i in out_used_cols]
    used_col_types = [col_typs[i] for i in out_used_cols]
    if index_column_name:
        used_col_names.append(index_column_name)
        assert isinstance(index_column_type, types.ArrayCompatible)
        used_col_types.append(index_column_type)

    return used_col_names, used_col_types


def prune_snowflake_select(
    used_col_names: list[str], pyarrow_schema: pa.Schema, converted_colnames: list[str]
) -> pa.Schema:
    """Prune snowflake select columns to only cover selected columns.

    Throws an error if the column is not found.
    """
    selected_fields = []
    for col_name in used_col_names:
        source_name = convert_col_name(col_name, converted_colnames)
        idx = pyarrow_schema.get_field_index(source_name)
        # If idx is -1, couldn't find a schema field with name `source_name`
        if idx < 0:
            raise BodoError(
                f"SQLReader Snowflake: Column {source_name} is not in source schema"
            )
        selected_fields.append(pyarrow_schema.field(idx))
    return pa.schema(selected_fields)


def _gen_snowflake_reader_chunked_py(
    col_names: list[str],
    col_typs: list[Any],
    index_column_name: Optional[str],
    index_column_type,
    out_used_cols: list[int],
    converted_colnames: list[str],
    db_type: str,
    limit: Optional[int],
    parallel: bool,
    typemap,
    filters: Optional[Any],
    pyarrow_schema: Optional[pa.Schema],
    is_dead_table: bool,
    is_select_query: bool,
    is_merge_into: bool,
    is_independent: bool,
    downcast_decimal_to_double: bool,
    chunksize: int,
):  # pragma: no cover
    """Function to generate main streaming SQL implementation.

    See _gen_sql_reader_py for argument documentation

    Args:
        chunksize: Number of rows in each batch
    """
    assert (
        db_type == "snowflake"
    ), f"Database {db_type} not supported in streaming IO mode, and should not go down this path"
    assert (
        pyarrow_schema is not None
    ), "SQLNode must contain a pyarrow_schema if reading from Snowflake"

    call_id = next_label()

    used_col_names, _ = prune_columns(
        col_names=col_names,
        col_typs=col_typs,
        out_used_cols=out_used_cols,
        index_column_name=index_column_name,
        index_column_type=index_column_type,
    )

    # Handle filter information because we may need to update the function header
    filter_args = ""  # TODO perhaps not needed

    func_text = f"def sql_reader_chunked_py(sql_request, conn, database_schema, {filter_args}):\n"

    if is_select_query:
        pyarrow_schema = prune_snowflake_select(
            used_col_names=used_col_names,
            pyarrow_schema=pyarrow_schema,
            converted_colnames=converted_colnames,
        )

    params: SnowflakeReadParams = SnowflakeReadParams.from_column_information(
        out_used_cols=out_used_cols,
        col_typs=col_typs,
        index_column_name=index_column_name,
        index_column_type=index_column_type,
    )

    func_text += "\n".join(
        [
            f"  total_rows_np = np.array([0], dtype=np.int64)",
            f"  snowflake_reader = snowflake_reader_init_py_entry(",
            f"    unicode_to_utf8(sql_request),",
            f"    unicode_to_utf8(conn),",
            f"    {parallel},",
            f"    {is_independent},",
            f"    pyarrow_schema_{call_id},",
            f"    {len(params.nullable_cols_array)},",
            f"    nullable_cols_array.ctypes,",
            f"    {len(params.snowflake_dict_cols_array)},",
            f"    snowflake_dict_cols_array.ctypes,",
            f"    total_rows_np.ctypes,",
            f"    {is_select_query and len(used_col_names) == 0},",
            f"    {is_select_query},",
            f"    {downcast_decimal_to_double},",
            f"    {chunksize},",
            f"    out_type,",
            f"  )",
            "",
        ]
    )
    func_text += "  return snowflake_reader"

    glbls = globals().copy()  # TODO: fix globals after Numba's #3355 is resolved
    glbls.update(
        np=np,
        unicode_to_utf8=unicode_to_utf8,
        snowflake_reader_init_py_entry=snowflake_reader_init_py_entry,
        out_type=ArrowReaderType(col_names, col_typs),
    )
    glbls.update(params._asdict())
    glbls.update({f"pyarrow_schema_{call_id}": pyarrow_schema})

    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    sql_reader_py = loc_vars["sql_reader_chunked_py"]

    # TODO: no_cpython_wrapper=True crashes for some reason
    jit_func = numba.njit(sql_reader_py)
    compiled_funcs.append(jit_func)
    return jit_func


def _gen_iceberg_reader_chunked_py(
    col_names: list[str],
    col_typs: list[Any],
    index_column_name: Optional[str],
    index_column_type,
    out_used_cols: list[int],
    converted_colnames: list[str],
    db_type: str,
    limit: Optional[int],
    parallel: bool,
    typemap,
    filters: Optional[Any],
    pyarrow_schema: Optional[pa.Schema],
    is_dead_table: bool,
    is_select_query: bool,
    is_merge_into: bool,
    is_independent: bool,
    downcast_decimal_to_double: bool,
    chunksize: int,
):  # pragma: no cover
    """Function to generate main streaming SQL implementation.

    See _gen_sql_reader_py for argument documentation

    Args:
        chunksize: Number of rows in each batch
    """
    assert (
        db_type == "iceberg"
    ), f"Database {db_type} not supported in streaming IO mode, and should not go down this path"
    assert (
        pyarrow_schema is not None
    ), "SQLNode must contain a pyarrow_schema if reading from Iceberg"

    call_id = next_label()

    # Handle filter information because we may need to update the function header
    filter_args = ""
    filter_map = {}
    filter_vars = []
    if filters:
        filter_map, filter_vars = bodo.ir.connector.generate_filter_map(filters)
        filter_args = ", ".join(filter_map.values())

    # Generate the partition filters and predicate filters. Note we pass
    # all col names as possible partitions via partition names.
    dnf_filter_str, expr_filter_str = bodo.ir.connector.generate_arrow_filters(
        filters,
        filter_map,
        filter_vars,
        col_names,
        col_names,
        col_typs,
        typemap,
        "iceberg",
    )

    # Determine selected C++ columns (and thus nullable) from original Iceberg
    # table / schema, assuming that Iceberg and Parquet field ordering is the same
    # Note that this does not include any locally generated columns (row id, file list, ...)
    # TODO: Update for schema evolution, when Iceberg Schema != Parquet Schema
    selected_cols: list[int] = [
        pyarrow_schema.get_field_index(col_names[i]) for i in out_used_cols
    ]
    nullable_cols = [
        int(is_nullable_ignore_sentinals(col_typs[i])) for i in selected_cols
    ]

    # pass indices to C++ of the selected string columns that are to be read
    # in dictionary-encoded format
    str_as_dict_cols = [
        i for i in selected_cols if col_typs[i] == bodo.dict_str_arr_type
    ]
    dict_str_cols_str = (
        f"dict_str_cols_arr_{call_id}.ctypes, np.int32({len(str_as_dict_cols)})"
        if str_as_dict_cols
        else "0, 0"
    )

    comma = "," if filter_args else ""
    func_text = (
        f"def sql_reader_chunked_py(sql_request, conn, database_schema, {filter_args}):\n"
        f"  ev = bodo.utils.tracing.Event('read_iceberg', {parallel})\n"
        f'  dnf_filters, expr_filters = get_filters_pyobject("{dnf_filter_str}", "{expr_filter_str}", ({filter_args}{comma}))\n'
        # Iceberg C++ Parquet Reader
        f"  iceberg_reader = iceberg_pq_reader_init_py_entry(\n"
        f"    unicode_to_utf8(conn),\n"
        f"    unicode_to_utf8(database_schema),\n"
        f"    unicode_to_utf8(sql_request),\n"
        f"    {parallel},\n"
        f"    {-1 if limit is None else limit},\n"
        f"    dnf_filters,\n"
        f"    expr_filters,\n"
        f"    selected_cols_arr_{call_id}.ctypes,\n"
        f"    {len(selected_cols)},\n"
        f"    nullable_cols_arr_{call_id}.ctypes,\n"
        f"    pyarrow_schema_{call_id},\n"
        f"    {dict_str_cols_str},\n"
        f"    {chunksize},\n"
        f"    out_type,\n"
        f"  )\n"
        f"  return iceberg_reader\n"
    )

    glbls = globals().copy()  # TODO: fix globals after Numba's #3355 is resolved
    glbls.update(
        {
            "unicode_to_utf8": unicode_to_utf8,
            "iceberg_pq_reader_init_py_entry": iceberg_pq_reader_init_py_entry,
            "get_filters_pyobject": bodo.io.parquet_pio.get_filters_pyobject,
            "out_type": ArrowReaderType(col_names, col_typs),
            f"selected_cols_arr_{call_id}": np.array(selected_cols, np.int32),
            f"nullable_cols_arr_{call_id}": np.array(nullable_cols, np.int32),
            f"dict_str_cols_arr_{call_id}": np.array(str_as_dict_cols, np.int32),
            f"pyarrow_schema_{call_id}": pyarrow_schema,
        }
    )

    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    sql_reader_py = loc_vars["sql_reader_chunked_py"]

    # TODO: no_cpython_wrapper=True crashes for some reason
    jit_func = numba.njit(sql_reader_py)
    compiled_funcs.append(jit_func)
    return jit_func


def _gen_sql_reader_py(
    col_names: list[str],
    col_typs: list[Any],
    index_column_name: Optional[str],
    index_column_type,
    out_used_cols: list[int],
    converted_colnames: list[str],
    db_type: str,
    limit: Optional[int],
    parallel: bool,
    typemap,
    filters: Optional[Any],
    pyarrow_schema: Optional[pa.Schema],
    is_dead_table: bool,
    is_select_query: bool,
    is_merge_into: bool,
    is_independent: bool,
    downcast_decimal_to_double: bool,
):
    """
    Function that generates the main SQL implementation. There are
    three main implementation paths:
        - Iceberg (calls parquet)
        - Snowflake (calls the Snowflake connector)
        - Regular SQL (uses SQLAlchemy)

    Args:
        col_names: Names of column output from the original query.
            This includes dead columns.
        col_typs: Types of column output from the original query.
            This includes dead columns.
        index_column_name: Name of column used as the index var or None
            if no column should be loaded.
        index_column_type: Type of column used as the index var or
            types.none if no column should be loaded.
        out_used_cols: List holding the values of columns that
            are live. For example if this is [0, 1, 3]
            it means all columns except for col_names[0],
            col_names[1], and col_names[3] are dead and
            should not be loaded (not including index).
        converted_colnames: List of column names that were modified from
            the original source name to match Pandas conventions. This is
            currently only used for Snowflake
        typingctx: Typing context used for compiling code.
        targetctx: Target context used for compiling code.
        db_type: Type of SQL source used to distinguish between backends.
        limit: Does the query contain a limit? This is only used to divide
            data with regular SQL.
        parallel: Is the implementation parallel?
        typemap: Maps variables name -> types. Used by iceberg for filters.
        filters: DNF Filter info used by iceberg to generate runtime filters.
            This should only be used for Iceberg.
        pyarrow_schema: PyArrow schema for the source table. This should only
            be used for Iceberg or Snowflake.
        is_select_query: Are we executing a select?
        is_merge_into: Does this query result from a merge into query? If so
            this limits the filtering we can do with Iceberg as we
            must load entire files.
    """
    # a unique int used to create global variables with unique names
    call_id = next_label()

    # Prune the columns to only those that are used.
    used_col_names, used_col_types = prune_columns(
        col_names=col_names,
        col_typs=col_typs,
        out_used_cols=out_used_cols,
        index_column_name=index_column_name,
        index_column_type=index_column_type,
    )

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
    py_table_type = types.none if is_dead_table else TableType(tuple(col_typs))

    # Handle filter information because we may need to update the function header
    filter_args = ""
    filter_map = {}
    filter_vars = []
    if filters and db_type == "iceberg":
        filter_map, filter_vars = bodo.ir.connector.generate_filter_map(filters)
        filter_args = ", ".join(filter_map.values())

    func_text = (
        f"def sql_reader_py(sql_request, conn, database_schema, {filter_args}):\n"
    )
    if db_type == "iceberg":
        assert (
            pyarrow_schema is not None
        ), "SQLNode must contain a pyarrow_schema if reading from an Iceberg database"

        # Generate the partition filters and predicate filters. Note we pass
        # all col names as possible partitions via partition names.
        dnf_filter_str, expr_filter_str = bodo.ir.connector.generate_arrow_filters(
            filters,
            filter_map,
            filter_vars,
            col_names,
            col_names,
            col_typs,
            typemap,
            "iceberg",
        )

        merge_into_row_id_col_idx = -1
        if is_merge_into and col_names.index("_BODO_ROW_ID") in out_used_cols:
            merge_into_row_id_col_idx = col_names.index("_BODO_ROW_ID")

        # Determine selected C++ columns (and thus nullable) from original Iceberg
        # table / schema, assuming that Iceberg and Parquet field ordering is the same
        # Note that this does not include any locally generated columns (row id, file list, ...)
        # TODO: Update for schema evolution, when Iceberg Schema != Parquet Schema
        selected_cols: list[int] = [
            pyarrow_schema.get_field_index(col_names[i])
            for i in out_used_cols
            if i != merge_into_row_id_col_idx
        ]
        selected_cols_map = {c: i for i, c in enumerate(selected_cols)}
        nullable_cols = [
            int(is_nullable_ignore_sentinals(col_typs[i])) for i in selected_cols
        ]

        # pass indices to C++ of the selected string columns that are to be read
        # in dictionary-encoded format
        str_as_dict_cols = [
            i for i in selected_cols if col_typs[i] == bodo.dict_str_arr_type
        ]
        dict_str_cols_str = (
            f"dict_str_cols_arr_{call_id}.ctypes, np.int32({len(str_as_dict_cols)})"
            if str_as_dict_cols
            else "0, 0"
        )

        comma = "," if filter_args else ""
        func_text += (
            f"  ev = bodo.utils.tracing.Event('read_iceberg', {parallel})\n"
            f'  dnf_filters, expr_filters = get_filters_pyobject("{dnf_filter_str}", "{expr_filter_str}", ({filter_args}{comma}))\n'
            # Iceberg C++ Parquet Reader
            f"  out_table, total_rows, file_list, snapshot_id = iceberg_pq_read_py_entry(\n"
            f"    unicode_to_utf8(conn),\n"
            f"    unicode_to_utf8(database_schema),\n"
            f"    unicode_to_utf8(sql_request),\n"
            f"    {parallel},\n"
            f"    {-1 if limit is None else limit},\n"
            f"    dnf_filters,\n"
            f"    expr_filters,\n"
            #     TODO Confirm that we're computing selected_cols correctly
            f"    selected_cols_arr_{call_id}.ctypes,\n"
            f"    {len(selected_cols)},\n"
            #     TODO Confirm that we're computing is_nullable correctly
            f"    nullable_cols_arr_{call_id}.ctypes,\n"
            f"    pyarrow_schema_{call_id},\n"
            f"    {dict_str_cols_str},\n"
            f"    {is_merge_into},\n"
            f"  )\n"
        )

        # Mostly copied over from _gen_pq_reader_py
        # TODO XXX Refactor?

        # Compute number of rows stored on rank for head optimization. See _gen_pq_reader_py
        if parallel:
            func_text += f"  local_rows = get_node_portion(total_rows, bodo.get_size(), bodo.get_rank())\n"
        else:
            func_text += f"  local_rows = total_rows\n"

        # Copied from _gen_pq_reader_py and simplified (no partitions or input_file_name)
        # table_idx is a list of index values for each array in the bodo.TableType being loaded from C++.
        # For a list column, the value is an integer which is the location of the column in the C++ Table.
        # Dead columns have the value -1.

        # For example if the Table Type is mapped like this: Table(arr0, arr1, arr2, arr3) and the
        # C++ representation is CPPTable(arr1, arr2), then table_idx = [-1, 0, 1, -1]

        # Note: By construction arrays will never be reordered (e.g. CPPTable(arr2, arr1)) in Iceberg
        # because we pass the col_names ordering.

        # If a table is dead we can skip the array for the table
        table_idx = None
        if not is_dead_table:
            table_idx = []
            j = 0
            for i in range(len(col_names)):
                # Should be same as from _gen_pq_reader_py
                # for i, col_num in enumerate(range(col_idxs)):
                # But we're assuming that the iceberg schema ordering is the same as the parquet ordering
                # TODO: Will change with schema evolution
                if j < len(out_used_cols) and i == out_used_cols[j]:
                    if i == merge_into_row_id_col_idx:
                        # row_id column goes at the end
                        table_idx.append(len(selected_cols))
                    else:
                        table_idx.append(selected_cols_map[i])
                    j += 1
                else:
                    table_idx.append(-1)
            table_idx = np.array(table_idx, dtype=np.int64)

        if is_dead_table:
            func_text += "  table_var = None\n"
        else:
            func_text += f"  table_var = cpp_table_to_py_table(out_table, table_idx_{call_id}, py_table_type_{call_id}, 0)\n"
            if len(out_used_cols) == 0:
                # Set the table length using the total rows if don't load any columns
                func_text += f"  table_var = set_table_len(table_var, local_rows)\n"

        # Handle index column
        index_var = "None"

        # Since we don't support `index_col`` with iceberg yet, we can't test this yet.
        if index_column_name is not None:  # pragma: no cover
            # The index column is defined by the SQLReader to always be placed at the end of the query.
            index_arr_ind = (len(out_used_cols) + 1) if not is_dead_table else 0
            index_var = (
                f"array_from_cpp_table(out_table, {index_arr_ind}, index_col_typ)"
            )

        func_text += f"  index_var = {index_var}\n"

        func_text += f"  delete_table(out_table)\n"
        func_text += f"  ev.finalize()\n"
        func_text += (
            "  return (total_rows, table_var, index_var, file_list, snapshot_id)\n"
        )

    elif db_type == "snowflake":  # pragma: no cover
        assert (
            pyarrow_schema is not None
        ), "SQLNode must contain a pyarrow_schema if reading from Snowflake"

        # Filter the schema by selected columns only
        # Only need to prune columns for SELECT queries
        if is_select_query:
            pyarrow_schema = prune_snowflake_select(
                used_col_names=used_col_names,
                pyarrow_schema=pyarrow_schema,
                converted_colnames=converted_colnames,
            )

        params: SnowflakeReadParams = SnowflakeReadParams.from_column_information(
            out_used_cols=out_used_cols,
            col_typs=col_typs,
            index_column_name=index_column_name,
            index_column_type=index_column_type,
        )
        # Track the total number of rows for loading 0 columns. If we load any
        # data this is garbage.
        func_text += (
            f"  ev = bodo.utils.tracing.Event('read_snowflake', {parallel})\n"
            f"  total_rows_np = np.array([0], dtype=np.int64)\n"
            f"  out_table = snowflake_read_py_entry(\n"
            f"    unicode_to_utf8(sql_request),\n"
            f"    unicode_to_utf8(conn),\n"
            f"    {parallel},\n"
            f"    {is_independent},\n"
            f"    pyarrow_schema_{call_id},\n"
            f"    {len(params.nullable_cols_array)},\n"
            f"    nullable_cols_array.ctypes,\n"
            f"    snowflake_dict_cols_array.ctypes,\n"
            f"    {len(params.snowflake_dict_cols_array)},\n"
            f"    total_rows_np.ctypes,\n"
            f"    {is_select_query and len(used_col_names) == 0},\n"
            f"    {is_select_query},\n"
            f"    {downcast_decimal_to_double},\n"
            f"  )\n"
            f"  check_and_propagate_cpp_exception()\n"
        )
        func_text += f"  total_rows = total_rows_np[0]\n"
        if parallel:
            func_text += f"  local_rows = get_node_portion(total_rows, bodo.get_size(), bodo.get_rank())\n"
        else:
            func_text += f"  local_rows = total_rows\n"
        if index_column_name:
            # The index is always placed in the last slot of the query if it exists.
            func_text += f"  index_var = array_from_cpp_table(out_table, {len(out_used_cols)}, index_col_typ)\n"
        else:
            # There is no index to load
            func_text += "  index_var = None\n"
        if is_dead_table:
            # We only load the index as the table is dead.
            func_text += "  table_var = None\n"
        else:
            # Map each logical column in the table to its location
            # in the input SQL table
            table_idx = map_cpp_to_py_table_column_idxs(
                col_names=col_names, out_used_cols=out_used_cols
            )
            func_text += f"  table_var = cpp_table_to_py_table(out_table, table_idx_{call_id}, py_table_type_{call_id}, 0)\n"
            if len(out_used_cols) == 0:
                if index_column_name:
                    # Set the table length using the index var if we load that column.
                    func_text += (
                        f"  table_var = set_table_len(table_var, len(index_var))\n"
                    )
                else:
                    # Set the table length using the total rows if don't load any columns
                    func_text += f"  table_var = set_table_len(table_var, local_rows)\n"
        func_text += "  delete_table(out_table)\n"
        func_text += "  ev.finalize()\n"
        func_text += "  return (total_rows, table_var, index_var, None, None)\n"

    else:
        if not is_dead_table:
            # Indicate which columns to load from the table
            func_text += f"  type_usecols_offsets_arr_{call_id}_2 = type_usecols_offsets_arr_{call_id}\n"
            type_usecols_offsets_arr = np.array(out_used_cols, dtype=np.int64)
        func_text += "  df_typeref_2 = df_typeref\n"
        func_text += "  sqlalchemy_check()\n"
        if db_type == "mysql":
            func_text += "  pymysql_check()\n"
        elif db_type == "oracle":
            func_text += "  cx_oracle_check()\n"
        elif db_type == "postgresql" or db_type == "postgresql+psycopg2":
            func_text += "  psycopg2_check()\n"

        if parallel:
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
                f"    index_var = df_ret.iloc[:, {len(out_used_cols)}].values\n"
            )
            func_text += f"    df_ret.drop(columns=df_ret.columns[{len(out_used_cols)}], inplace=True)\n"
        else:
            # Dead Index
            func_text += "    index_var = None\n"
        if not is_dead_table:
            func_text += f"    arrs = []\n"
            func_text += f"    for i in range(df_ret.shape[1]):\n"
            func_text += f"      arrs.append(df_ret.iloc[:, i].values)\n"
            # Bodo preserves all of the original types needed at typing in col_typs
            func_text += f"    table_var = Table(arrs, type_usecols_offsets_arr_{call_id}_2, {len(col_names)})\n"
        else:
            # Dead Table
            func_text += "    table_var = None\n"
        func_text += "  return (-1, table_var, index_var, None, None)\n"

    glbls = globals()  # TODO: fix globals after Numba's #3355 is resolved
    glbls.update(
        {
            "bodo": bodo,
            f"py_table_type_{call_id}": py_table_type,
            "index_col_typ": index_column_type,
        }
    )
    if db_type in ("iceberg", "snowflake"):
        glbls.update(
            {
                f"table_idx_{call_id}": table_idx,
                f"pyarrow_schema_{call_id}": pyarrow_schema,
                "unicode_to_utf8": unicode_to_utf8,
                "check_and_propagate_cpp_exception": check_and_propagate_cpp_exception,
                "array_from_cpp_table": array_from_cpp_table,
                "delete_table": delete_table,
                "cpp_table_to_py_table": cpp_table_to_py_table,
                "set_table_len": bodo.hiframes.table.set_table_len,
                "get_node_portion": bodo.libs.distributed_api.get_node_portion,
            }
        )

    if db_type == "iceberg":
        glbls.update(
            {
                # TODO: Remove type ignores when refactoring Iceberg read code generation
                # out of this function
                f"selected_cols_arr_{call_id}": np.array(selected_cols, np.int32),  # type: ignore
                f"nullable_cols_arr_{call_id}": np.array(nullable_cols, np.int32),  # type: ignore
                f"dict_str_cols_arr_{call_id}": np.array(str_as_dict_cols, np.int32),  # type: ignore
                f"py_table_type_{call_id}": py_table_type,
                "get_filters_pyobject": bodo.io.parquet_pio.get_filters_pyobject,
                "iceberg_pq_read_py_entry": iceberg_pq_read_py_entry,
            }
        )
    elif db_type == "snowflake":
        glbls.update(
            {
                "np": np,
                "snowflake_read_py_entry": _snowflake_read,
                "nullable_cols_array": params.nullable_cols_array,
                "snowflake_dict_cols_array": params.snowflake_dict_cols_array,
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


parquet_predicate_type = ParquetPredicateType()


@intrinsic(prefer_literal=True)
def iceberg_pq_read_py_entry(
    typingctx,
    conn_str,
    db_schema,
    sql_request_str,
    parallel,
    limit,
    dnf_filters,
    expr_filters,
    selected_cols,
    num_selected_cols,
    nullable_cols,
    pyarrow_schema,
    dict_encoded_cols,
    num_dict_encoded_cols,
    is_merge_into_cow,
):
    """Perform a read from an Iceberg Table using a the C++
    iceberg_pq_read_py_entry function. That function returns a C++ Table
    and updates 3 pointers:
        - The number of rows read
        - A PyObject which is a list of relative paths to file names (used in merge).
          If unused this will be None.
        - An int64 for the snapshot id (used in merge). If unused this will be -1.

    The llvm code then packs these results into an Output tuple with the following types
        (C++Table, int64, pyobject_of_list_type, int64)

    pyobject_of_list_type is a wrapper type around a Pyobject that enables reference counting
    to avoid memory leaks.

    Args:
        typingctx (Context): Context used for typing
        conn_str (types.voidptr): C string for the connection
        db_schema (types.voidptr): C string for the db_schema
        sql_request_str (types.voidptr): C string for sql request
        parallel (types.boolean): Is the read in parallel
        limit (types.int64): Max number of rows to read. -1 if all rows
        dnf_filters (parquet_predicate_type): PyObject for DNF filters.
        expr_filters (parquet_predicate_type): PyObject for Expr filters
        selected_cols (types.voidptr): C pointers of integers for selected columns
        num_selected_cols (types.int64): Length of selected_cols
        nullable_cols (types.voidptr): C pointers of 0 or 1 for if each selected column is nullable
        pyarrow_schema (pyarrow_schema_type): Pyobject with the pyarrow schema for the output.
        dict_encoded_cols (types.voidptr): Array fo column numbers that are dictionary encoded.
        num_dict_encoded_cols (_type_): Length of dict_encoded_cols
        is_merge_into_cow (bool): Are we doing a merge?
    """

    def codegen(context, builder, signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(1),  # bool
                lir.IntType(64),  # int64
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(32),  # int32
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(32),  # int32
                lir.IntType(1),  # bool
                lir.IntType(64).as_pointer(),  # int64_t*
                lir.IntType(8).as_pointer().as_pointer(),  # PyObject**
                lir.IntType(64).as_pointer(),  # int64_t*
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="iceberg_pq_read_py_entry"
        )
        # Allocate the pointers to update
        num_rows_ptr = cgutils.alloca_once(builder, lir.IntType(64))
        file_list_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
        snapshot_id_ptr = cgutils.alloca_once(builder, lir.IntType(64))
        total_args = args + (num_rows_ptr, file_list_ptr, snapshot_id_ptr)
        table = builder.call(fn_tp, total_args)
        # Check for C++ errors
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        # Convert the file_list to underlying struct
        file_list_pyobj = builder.load(file_list_ptr)
        file_list_struct = cgutils.create_struct_proxy(types.pyobject_of_list_type)(
            context, builder
        )
        pyapi = context.get_python_api(builder)
        # borrows and manages a reference for obj (see comments in py_objs.py)
        file_list_struct.meminfo = pyapi.nrt_meminfo_new_from_pyobject(
            context.get_constant_null(types.voidptr), file_list_pyobj
        )
        file_list_struct.pyobj = file_list_pyobj
        # `nrt_meminfo_new_from_pyobject` increfs the object (holds a reference)
        # so need to decref since the object is not live anywhere else.
        pyapi.decref(file_list_pyobj)

        # Fetch the underlying data from the pointers.
        items = [
            table,
            builder.load(num_rows_ptr),
            file_list_struct._getvalue(),
            builder.load(snapshot_id_ptr),
        ]
        # Return the tuple
        return context.make_tuple(builder, ret_type, items)

    ret_type = types.Tuple(
        [table_type, types.int64, types.pyobject_of_list_type, types.int64]
    )
    sig = ret_type(
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.boolean,
        types.int64,
        parquet_predicate_type,  # dnf filters
        parquet_predicate_type,  # expr filters
        types.voidptr,
        types.int32,
        types.voidptr,
        pyarrow_schema_type,
        types.voidptr,
        types.int32,
        types.boolean,
    )
    return sig, codegen


@intrinsic(prefer_literal=True)
def iceberg_pq_reader_init_py_entry(
    typingctx,
    conn_str,
    db_schema,
    sql_request_str,
    parallel,
    limit,
    dnf_filters,
    expr_filters,
    selected_cols,
    num_selected_cols,
    nullable_cols,
    pyarrow_schema,
    dict_encoded_cols,
    num_dict_encoded_cols,
    chunksize_t,
    arrow_reader_t,
):
    """Construct a reader for an Iceberg Table using a the C++
    iceberg_pq_reader_init_py_entry function. That function returns an ArrowReader

    Args:
        typingctx (Context): Context used for typing
        conn_str (types.voidptr): C string for the connection
        db_schema (types.voidptr): C string for the db_schema
        sql_request_str (types.voidptr): C string for sql request
        parallel (types.boolean): Is the read in parallel
        limit (types.int64): Max number of rows to read. -1 if all rows
        dnf_filters (parquet_predicate_type): PyObject for DNF filters.
        expr_filters (parquet_predicate_type): PyObject for Expr filters
        selected_cols (types.voidptr): C pointers of integers for selected columns
        num_selected_cols (types.int64): Length of selected_cols
        nullable_cols (types.voidptr): C pointers of 0 or 1 for if each selected column is nullable
        pyarrow_schema (pyarrow_schema_type): Pyobject with the pyarrow schema for the output.
        dict_encoded_cols (types.voidptr): Array fo column numbers that are dictionary encoded.
        num_dict_encoded_cols (_type_): Length of dict_encoded_cols
        arrow_reader_t (ArrowReader): The typing of the output ArrowReader
    """

    assert isinstance(arrow_reader_t, types.TypeRef) and isinstance(
        arrow_reader_t.instance_type, ArrowReaderType
    ), "iceberg_pq_reader_init_py_entry(): The last argument arrow_reader must by a TypeRef to an ArrowReader"

    def codegen(context: "BaseContext", builder: "IRBuilder", signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),  # table_name void*
                lir.IntType(8).as_pointer(),  # conn_str void*
                lir.IntType(8).as_pointer(),  # schema void*
                lir.IntType(1),  # parallel bool
                lir.IntType(64),  # tot_rows_to_read int64
                lir.IntType(8).as_pointer(),  # dnf_filters void*
                lir.IntType(8).as_pointer(),  # expr_filters void*
                lir.IntType(8).as_pointer(),  # _selected_fields void*
                lir.IntType(32),  # num_selected_fields int32
                lir.IntType(8).as_pointer(),  # _is_nullable void*
                lir.IntType(8).as_pointer(),  # pyarrow_schema void*
                lir.IntType(8).as_pointer(),  # _str_as_dict_cols void*
                lir.IntType(32),  # num_str_as_dict_cols int32
                lir.IntType(64),  # chunksize int64
            ],
        )

        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="iceberg_pq_reader_init_py_entry"
        )

        iceberg_reader = builder.call(fn_tp, args[:-1])
        inlined_check_and_propagate_cpp_exception(context, builder)
        return iceberg_reader

    sig = arrow_reader_t.instance_type(
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.boolean,
        types.int64,
        parquet_predicate_type,  # dnf filters
        parquet_predicate_type,  # expr filters
        types.voidptr,
        types.int32,
        types.voidptr,
        pyarrow_schema_type,  # pyarrow_schema
        types.voidptr,  # str_as_dict_cols
        types.int32,  # num_str_as_dict_cols
        types.int64,  # chunksize
        arrow_reader_t,  # typing only
    )
    return sig, codegen


_snowflake_read = types.ExternalFunction(
    "snowflake_read_py_entry",
    table_type(
        types.voidptr,  # query
        types.voidptr,  # conn_str
        types.boolean,  # parallel
        types.boolean,  # is_independent
        pyarrow_schema_type,
        types.int64,  # n_fields
        types.voidptr,  # _is_nullable
        types.voidptr,  # _str_as_dict_cols
        types.int32,  # num_str_as_dict_cols
        types.voidptr,  # total_nrows
        types.boolean,  # _only_length_query
        types.boolean,  # _is_select_query
        types.boolean,  # downcast_decimal_to_double
    ),
)


@intrinsic(prefer_literal=True)
def snowflake_reader_init_py_entry(
    typingctx,
    query_t,
    conn_t,
    parallel_t,
    is_independent_t,
    pyarrow_schema_t,
    n_fields_t,
    is_nullable_t,
    num_str_as_dict_cols_t,
    str_as_dict_cols_t,
    total_nrows_t,
    only_length_query_t,
    is_select_query_t,
    downcast_decimal_to_double_t,
    chunksize_t,
    arrow_reader_t,
):  # pragma: no cover
    assert isinstance(arrow_reader_t, types.TypeRef) and isinstance(
        arrow_reader_t.instance_type, ArrowReaderType
    ), "snowflake_reader_init_py_entry(): The last argument arrow_reader must by a TypeRef to an ArrowReader"
    assert (
        pyarrow_schema_t == pyarrow_schema_type
    ), "snowflake_reader_init_py_entry(): The 5th argument pyarrow_schema must by a PyArrow schema"

    def codegen(context: "BaseContext", builder: "IRBuilder", signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),  # query void*
                lir.IntType(8).as_pointer(),  # conn_str void*
                lir.IntType(1),  # parallel bool
                lir.IntType(1),  # is_independent bool
                lir.IntType(8).as_pointer(),  # pyarrow_schema PyObject*
                lir.IntType(64),  # n_fields int64
                lir.IntType(8).as_pointer(),  # _is_nullable void*
                lir.IntType(32),  # num_str_as_dict_cols int32
                lir.IntType(8).as_pointer(),  # _str_as_dict_cols void*
                lir.IntType(8).as_pointer(),  # total_nrows void*
                lir.IntType(1),  # _only_length_query bool
                lir.IntType(1),  # _is_select_query bool
                lir.IntType(1),  # downcast_decimal_to_double
                lir.IntType(64),  # chunksize
            ],
        )

        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="snowflake_reader_init_py_entry"
        )

        snowflake_reader = builder.call(fn_tp, args[:-1])
        inlined_check_and_propagate_cpp_exception(context, builder)
        return snowflake_reader

    sig = arrow_reader_t.instance_type(
        types.voidptr,  # query
        types.voidptr,  # conn_str
        types.boolean,  # parallel
        types.boolean,  # is_independent
        pyarrow_schema_type,
        types.int64,  # n_fields
        types.voidptr,  # _is_nullable
        types.int32,  # num_str_as_dict_cols
        types.voidptr,  # _str_as_dict_cols
        types.voidptr,  # total_nrows
        types.boolean,  # _only_length_query
        types.boolean,  # _is_select_query
        types.boolean,  # downcast_decimal_to_double
        types.int64,  # chunksize
        arrow_reader_t,  # typing only
    )
    return sig, codegen
