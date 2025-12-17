"""
Implementation of pd.read_sql in Bodo.
We piggyback on the pandas implementation. Future plan is to have a faster
version for this task.
"""

from __future__ import annotations

import datetime
import sys
from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
)

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
from numba.extending import (
    intrinsic,
    overload,
)

import bodo
import bodo.ir.connector
import bodo.ir.filter as bif
import bodo.user_logging
from bodo.ext import stream_join_cpp
from bodo.hiframes.table import Table, TableType
from bodo.io import arrow_cpp  # type: ignore
from bodo.io.arrow_reader import ArrowReaderType
from bodo.io.helpers import map_cpp_to_py_table_column_idxs, pyarrow_schema_type
from bodo.ir.connector import Connector
from bodo.ir.filter import Filter, supported_funcs_map
from bodo.libs.array import (
    array_from_cpp_table,
    cpp_table_to_py_table,
    delete_table,
    table_type,
)
from bodo.libs.dict_arr_ext import dict_str_arr_type
from bodo.libs.distributed_api import bcast_scalar
from bodo.libs.str_ext import string_type, unicode_to_utf8
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.transforms.distributed_analysis import Distribution
from bodo.transforms.table_column_del_pass import (
    ir_extension_table_column_use,
    remove_dead_column_extensions,
)
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.typing import (
    BodoError,
    get_overload_const_str,
    is_nullable_ignore_sentinels,
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
    "get_runtime_join_filter_min_max_py_entrypt",
    stream_join_cpp.get_runtime_join_filter_min_max_py_entrypt,
)
ll.add_symbol(
    "is_empty_build_table_py_entrypt",
    stream_join_cpp.is_empty_build_table_py_entrypt,
)
ll.add_symbol(
    "has_runtime_join_filter_unique_values_py_entrypt",
    stream_join_cpp.has_runtime_join_filter_unique_values_py_entrypt,
)
ll.add_symbol(
    "get_runtime_join_filter_unique_values_py_entrypt",
    stream_join_cpp.get_runtime_join_filter_unique_values_py_entrypt,
)

DEFAULT_ROOT = 0


class SnowflakeReadParams(NamedTuple):
    """Common inputs into snowflake reader functions."""

    snowflake_dict_cols_array: npt.NDArray[np.int32]
    nullable_cols_array: npt.NDArray[np.int32]

    @classmethod
    def from_column_information(
        cls,
        out_used_cols: list[int],
        col_typs: list[types.ArrayCompatible],
        index_column_name: str | None,
        index_column_type: types.ArrayCompatible | types.NoneType,
    ):  # pragma: no cover
        """Construct a SnowflakeReaderParams from the IR parameters"""
        col_indices_map = {c: i for i, c in enumerate(out_used_cols)}
        snowflake_dict_cols = [
            col_indices_map[i]
            for i in out_used_cols
            if col_typs[i] == dict_str_arr_type
        ]

        nullable_cols = [
            int(is_nullable_ignore_sentinels(col_typs[i])) for i in out_used_cols
        ]
        # Handle if we need to append an index
        if index_column_name:
            nullable_cols.append(int(is_nullable_ignore_sentinels(index_column_type)))
        snowflake_dict_cols_array = np.array(snowflake_dict_cols, dtype=np.int32)
        nullable_cols_array = np.array(nullable_cols, dtype=np.int32)

        return cls(
            snowflake_dict_cols_array=snowflake_dict_cols_array,
            nullable_cols_array=nullable_cols_array,
        )


class SqlReader(Connector):
    connector_typ: str = "sql"
    filters: Filter | None

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
        index_column_name: str | None,
        index_column_type: types.ArrayCompatible | types.NoneType,
        database_schema: str | None,
        # Only relevant for Snowflake
        pyarrow_schema: pa.Schema | None,
        # Runtime should downcast decimal columns to double
        # Only relevant for Snowflake ATM
        downcast_decimal_to_double: bool,
        # Batch size to read chunks in, or none, to read the entire table together
        # Only supported for Snowflake
        # Treated as compile-time constant for simplicity
        # But not enforced that all chunks are this size
        chunksize: int | None = None,
        # Operator ID generated by BodoSQL for query profile
        # purposes. Only supported in the streaming Snowflake
        # Read case.
        sql_op_id: int = -1,
        # List of tuples representing runtime join filters
        # that have been pushed down to I/O.
        rtjf_terms: list[tuple[ir.Var, tuple[int], tuple[int, int, str]]] | None = None,
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
        # Only relevant for Snowflake
        self.pyarrow_schema = pyarrow_schema
        # Is the variable currently alive. This should be replaced with more
        # robust handling in connectors.
        self.is_live_table = True

        self.downcast_decimal_to_double = downcast_decimal_to_double
        self.chunksize = chunksize
        self.sql_op_id = sql_op_id

        self.rtjf_terms = rtjf_terms

    def __repr__(self) -> str:  # pragma: no cover
        out_varnames = tuple(v.name for v in self.out_vars)
        runtime_join_filters = rtjf_term_repr(self.rtjf_terms)
        return f"{out_varnames} = SQLReader(sql_request={self.sql_request}, connection={self.connection}, out_col_names={self.out_table_col_names}, out_col_types={self.out_table_col_types}, df_out_varname={self.df_out_varname}, limit={self.limit}, unsupported_columns={self.unsupported_columns}, unsupported_arrow_types={self.unsupported_arrow_types}, is_select_query={self.is_select_query}, index_column_name={self.index_column_name}, index_column_type={self.index_column_type}, out_used_cols={self.out_used_cols}, database_schema={self.database_schema}, pyarrow_schema={self.pyarrow_schema}, downcast_decimal_to_double={self.downcast_decimal_to_double}, sql_op_id={self.sql_op_id}, runtime_join_filters={runtime_join_filters})"

    def out_vars_and_types(self) -> list[tuple[str, types.Type]]:
        if self.is_streaming:
            return [
                (
                    self.out_vars[0].name,
                    ArrowReaderType(self.out_table_col_names, self.out_table_col_types),
                )
            ]
        return [
            (self.out_vars[0].name, TableType(tuple(self.out_table_col_types))),
            (self.out_vars[1].name, self.index_column_type),
        ]

    def out_table_distribution(self) -> Distribution:
        if not self.is_select_query:
            return Distribution.REP
        elif self.limit is not None:
            return Distribution.OneD_Var
        else:
            return Distribution.OneD


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
    if (
        not sql_node.has_side_effects
        and table_var not in lives
        and index_var not in lives
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

    return sql_node


class SnowflakeFilterVisitor(bif.FilterVisitor[str]):
    """
    Bodo IR Filter Visitor to construct a SQL WHERE clause from a filter.
    Used for DB & Snowflake filter pushdown.

    Args:
        filter_map: A mapping of IR names to the compiler filler names
        converted_colnames: A list of column names that have been converted
        typemap: A mapping of variable names to their types

    Returns:
        A string representing the SQL WHERE clause
    """

    def __init__(self, filter_map, converted_colnames, typemap) -> None:
        self.filter_map = filter_map
        self.converted_colnames = converted_colnames
        self.typemap = typemap

    def visit_scalar(self, scalar: bif.Scalar) -> str:
        scalar_name = scalar.val.name
        return f"{{{self.filter_map[scalar_name]}}}"

    def visit_ref(self, ref: bif.Ref) -> str:
        col_name = convert_col_name(ref.val, self.converted_colnames)
        return '\\"' + col_name + '\\"'

    def visit_op(self, op: bif.Op) -> str:
        if op.op == "ALWAYS_TRUE":
            # Special operator for True
            return "(TRUE)"
        elif op.op == "ALWAYS_FALSE":
            # Special operators for False.
            return "(FALSE)"
        elif op.op == "ALWAYS_NULL":
            # Special operators for NULL.
            return "(NULL)"

        elif op.op == "AND":
            return " AND ".join(self.visit(c) for c in op.args)
        elif op.op == "OR":
            return " OR ".join(self.visit(c) for c in op.args)
        elif op.op == "NOT":
            return f"(NOT {self.visit(op.args[0])})"

        elif op.op == "IS_NULL":
            return f"({self.visit(op.args[0])} IS NULL)"
        elif op.op == "IS_NOT_NULL":
            return f"({self.visit(op.args[0])} IS NOT NULL)"

        elif op.op == "case_insensitive_equality":
            # Equality is just =, not a function
            return (
                f"(LOWER({self.visit(op.args[0])}) = LOWER({self.visit(op.args[1])}))"
            )
        elif op.op in (
            "case_insensitive_startswith",
            "case_insensitive_endswith",
            "case_insensitive_contains",
        ):
            op_name = op.op[len("case_insensitive_") :]
            return f"({op_name}(LOWER({self.visit(op.args[0])}), LOWER({self.visit(op.args[1])})))"
        elif op.op in ("like", "ilike"):
            # You can't pass the empty string to escape. As a result we
            # must confirm its not the empty string
            escape_arg = op.args[2]
            assert isinstance(escape_arg, bif.Scalar)
            has_escape = True
            escape_typ = self.typemap[escape_arg.val.name]
            if is_overload_constant_str(escape_typ):
                escape_val = get_overload_const_str(escape_typ)
                has_escape = escape_val != ""
            escape_section = f"escape {self.visit(escape_arg)}" if has_escape else ""

            return f"({self.visit(op.args[0])} {op.op} {self.visit(op.args[1])} {escape_section})"

        # Infix Operators
        elif op.op in ("=", "==", "!=", "<>", "<", "<=", ">", ">=", "IN"):
            return f"({self.visit(op.args[0])} {op.op} {self.visit(op.args[1])})"

        # Handles all functions in general, including previous special cases like
        # REGEXP_LIKE
        else:
            func = op.op
            if func not in supported_funcs_map:
                raise NotImplementedError(
                    f"Snowflake Filter pushdown not implemented for {func} function"
                )
            sql_func = supported_funcs_map[func]
            return f"({sql_func}({', '.join(self.visit(c) for c in op.args)}))"


# Class for a the RTJF min/max/unique stored values
this_module = sys.modules[__name__]
RtjfValueType, _ = install_py_obj_class(
    types_name="rtjf_value_type",
    python_type=None,
    module=this_module,
    class_name="RtjfValueType",
    model_name="RtjfValueType",
)


@intrinsic
def get_runtime_join_filter_min_max(
    typingctx, state_var_t, key_index_t, is_min_t, precision_t
):
    """
    Fetches the minimum or maximum value from a runtime join filter corresponding
    to the specified key index, if one exists. Returns as a PyObject that is
    either the string representation of the value, or None if it cannot be found.
    Whether the min or max is returned depends on the is_min argument.
    """
    assert isinstance(state_var_t, bodo.libs.streaming.join.JoinStateType)

    def codegen(context, builder, signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),  # join state
                lir.IntType(64),  # key_index
                lir.IntType(1),  # is_min
                lir.IntType(64),  # precision
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="get_runtime_join_filter_min_max_py_entrypt"
        )
        rtjf_min = builder.call(fn_tp, args)
        rtjf_min_struct = cgutils.create_struct_proxy(types.rtjf_value_type)(
            context, builder
        )
        pyapi = context.get_python_api(builder)
        # borrows and manages a reference for obj (see comments in py_objs.py)
        rtjf_min_struct.meminfo = pyapi.nrt_meminfo_new_from_pyobject(
            context.get_constant_null(types.voidptr), rtjf_min
        )
        rtjf_min_struct.pyobj = rtjf_min
        inlined_check_and_propagate_cpp_exception(context, builder)
        return rtjf_min_struct._getvalue()

    sig = types.rtjf_value_type(state_var_t, key_index_t, is_min_t, precision_t)
    return sig, codegen


@intrinsic
def is_empty_build_table(typingctx, state_var_t):
    """
    Returns if a join state has a completely empty build table, which can
    be used to prune the entire probe table.
    """
    assert isinstance(state_var_t, bodo.libs.streaming.join.JoinStateType)

    def codegen(context, builder, signature, args):
        fnty = lir.FunctionType(
            lir.IntType(1),
            [
                lir.IntType(8).as_pointer(),  # join state
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module,
            fnty,
            name="is_empty_build_table_py_entrypt",
        )
        ret = builder.call(fn_tp, args)
        inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.bool_(state_var_t)
    return sig, codegen


@intrinsic
def has_runtime_join_filter_unique_values(typingctx, state_var_t, key_index_t):
    """
    Returns whether a join state has a list of unique values for a specific
    join key column.
    """
    assert isinstance(state_var_t, bodo.libs.streaming.join.JoinStateType)

    def codegen(context, builder, signature, args):
        fnty = lir.FunctionType(
            lir.IntType(1),
            [
                lir.IntType(8).as_pointer(),  # join state
                lir.IntType(64),  # key_index_t
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module,
            fnty,
            name="has_runtime_join_filter_unique_values_py_entrypt",
        )
        ret = builder.call(fn_tp, args)
        inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = types.bool_(state_var_t, key_index_t)
    return sig, codegen


@intrinsic
def get_runtime_join_filter_unique_values(typingctx, state_var_t, key_index_t):
    """
    Returns whether the list of unique values from the join state corresponding
    to a certain key column.
    """
    assert isinstance(state_var_t, bodo.libs.streaming.join.JoinStateType)

    def codegen(context, builder, signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),  # join state
                lir.IntType(64),  # key_index_t
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module,
            fnty,
            name="get_runtime_join_filter_unique_values_py_entrypt",
        )
        unique_vals = builder.call(fn_tp, args)
        unique_vals_struct = cgutils.create_struct_proxy(types.rtjf_value_type)(
            context, builder
        )
        pyapi = context.get_python_api(builder)
        # borrows and manages a reference for obj (see comments in py_objs.py)
        unique_vals_struct.meminfo = pyapi.nrt_meminfo_new_from_pyobject(
            context.get_constant_null(types.voidptr), unique_vals
        )
        unique_vals_struct.pyobj = unique_vals
        inlined_check_and_propagate_cpp_exception(context, builder)
        return unique_vals_struct._getvalue()

    sig = types.rtjf_value_type(state_var_t, key_index_t)
    return sig, codegen


def convert_pyobj_to_snowflake_str(pyobj, time_zone):
    """
    Converts a Python object to the equivalent Snowflake
    representation that can be injected as a string into
    query text. For example:

    42 -> '42'
    "foo bar" -> "'foo bar'"
    datetime.date(2024, 3, 14) -> "DATE '2024-03-14'"
    bodo.types.Time(12, 30, 59, 0, 0, 99) -> "TIME_FROM_PARTS(0, 0, 0, 45059000000091)"
    pd.Timestamp("2024-07-04 12:30:01.025601") -> "TIMESTAMP_FROM_PARTS(2024, 7, 4, 12, 30, 1, 25601000)"
    """
    if isinstance(pyobj, str):
        return f"'{pyobj}'"
    elif isinstance(pyobj, pd.Timestamp):
        if time_zone is None:
            suffix = "_NTZ"
        else:
            suffix = "_LTZ"
            # Convert the Timestamp from a UTC value to an equivalent value in the desired timezone,
            # thus allowing us to get the correct value of each date/time component in the desired timezone.
            pyobj = pyobj.tz_localize("UTC").tz_convert(time_zone)
        return f"TIMESTAMP{suffix}_FROM_PARTS({pyobj.year}, {pyobj.month}, {pyobj.day}, {pyobj.hour}, {pyobj.minute}, {pyobj.second}, {pyobj.value % 1_000_000_000})"
    elif isinstance(pyobj, datetime.date):
        return f"DATE '{pyobj.year:04}-{pyobj.month:02}-{pyobj.day:02}'"
    elif isinstance(pyobj, bodo.types.Time):
        return f"TIME_FROM_PARTS(0, 0, 0, {pyobj.value})"
    else:
        return str(pyobj)


def gen_runtime_join_filter_cond(state_var, col_indices, precisions, time_zones):
    pass


@overload(gen_runtime_join_filter_cond)
def overload_gen_runtime_join_filter_cond(
    state_var, col_indices, precisions, time_zones
):
    """
    Takes in a join state and tuple of column indices and uses them to construct new entries
    to conditions based on runtime join filter state (e.g. build table size, min/max, magic sets).
    The output adds new strings to the conditions list in snowflake format.
    """

    def impl(state_var, col_indices, precisions, time_zones):  # pragma: no cover
        is_empty = is_empty_build_table(state_var)
        if is_empty:
            # If the build table is empty in an inner hash join, then
            # we can't match any rows, so return a FALSE condition
            # to tell Snowflake to skip every row.
            return ["FALSE /* Empty Build Table in a Join */"]
        else:
            local_conds = []
            n_cols = len(col_indices)
            for i in range(n_cols):
                col_idx = col_indices[i]
                if col_idx == -1:
                    continue

                included_low_ndv_filter = False
                # If it exists, get the list of unique values from the join state
                if has_runtime_join_filter_unique_values(state_var, np.int64(i)):
                    unique_values = get_runtime_join_filter_unique_values(
                        state_var, np.int64(i)
                    )
                    # Use object mode to convert to a string representation of the containment check
                    with bodo.ir.object_mode.no_warning_objmode(
                        unique_as_strings="list_str_type"
                    ):
                        unique_as_strings = []
                        for elem in unique_values:
                            unique_as_strings.append(
                                convert_pyobj_to_snowflake_str(elem, time_zones[i])
                            )

                    if len(unique_as_strings) > 0:
                        local_conds.append(
                            f"(${col_idx + 1} IN ({', '.join(sorted(unique_as_strings))}))"
                        )
                        included_low_ndv_filter = True

                if not included_low_ndv_filter:
                    # Otherwise, get the min/max value bounds for the current key column
                    # from the join state
                    min_val = get_runtime_join_filter_min_max(
                        state_var, np.int64(i), True, precisions[i]
                    )
                    max_val = get_runtime_join_filter_min_max(
                        state_var, np.int64(i), False, precisions[i]
                    )
                    # Use object mode to convert to a string representation of the bounds check
                    with bodo.ir.object_mode.no_warning_objmode(
                        min_result="unicode_type", max_result="unicode_type"
                    ):
                        min_result = max_result = ""
                        if min_val is not None:
                            min_result = f"(${col_idx + 1} >= {convert_pyobj_to_snowflake_str(min_val, time_zones[i])})"
                        if max_val is not None:
                            max_result = f"(${col_idx + 1} <= {convert_pyobj_to_snowflake_str(max_val, time_zones[i])})"
                    # If the results were successful, add to the conjunction list
                    if min_result != "":
                        local_conds.append(min_result)
                    if max_result != "":
                        local_conds.append(max_result)
        return local_conds

    return impl


def gen_runtime_join_filter_interval_cond(
    state_var, probe_cols, build_cols, ops, precisions, time_zones
):
    pass


@overload(gen_runtime_join_filter_interval_cond)
def overload_gen_runtime_join_filter_interval_cond(
    state_var, probe_cols, build_cols, ops, precisions, time_zones
):
    """
    Implements filters generated by interval joins. The probe_cols are filtered in snowflake if the corresponding
    build column has a min/max value, depending on the operators.
    """

    def impl(
        state_var, probe_cols, build_cols, ops, precisions, time_zones
    ):  # pragma: no cover
        local_conds = []
        n = len(probe_cols)
        for i in range(n):
            build_col = build_cols[i]
            probe_col = probe_cols[i]
            op = ops[i]
            min_val = get_runtime_join_filter_min_max(
                state_var, build_col, True, precisions[i]
            )
            max_val = get_runtime_join_filter_min_max(
                state_var, build_col, False, precisions[i]
            )
            with bodo.ir.object_mode.no_warning_objmode(result="unicode_type"):
                result = ""
                if op in (">", ">=") and min_val is not None:
                    result = f"(${probe_col + 1} {op} {convert_pyobj_to_snowflake_str(min_val, time_zones[i])})"
                elif op in ("<", "<=") and max_val is not None:
                    result = f"(${probe_col + 1} {op} {convert_pyobj_to_snowflake_str(max_val, time_zones[i])})"
            # If the results were successful, add to the conjunction list
            if result != "":
                local_conds.append(result)
        return local_conds

    return impl


def rtjf_term_repr(rtjf_terms):
    """
    Converts a list of runtime join filter terms to a string representation.
    """
    return (
        "None"
        if rtjf_terms is None
        else "["
        + ", ".join(
            f"({state_var.name}, {col_indices})"
            for state_var, col_indices, _ in rtjf_terms
        )
        + "]"
    )


def extract_rtjf_terms(
    rtjf_terms: list[tuple[ir.Var, tuple[int], tuple[int, int, str]]],
) -> tuple[list[tuple[int]], list[ir.Var], list[str], list[tuple[int, int, str]]]:
    """
    Extracts the runtime join filter terms into separate lists for the column indices,
    state variables, and state variable names. For state variable names, the names are
    reformatted to avoid IR variable characters like "$".
    """
    rtjf_state_col_indices = []
    rtjf_state_args = []
    rtjf_state_names = []
    rtjf_non_equality_conditions = []
    for state_var, col_indices, non_equality_conditions in rtjf_terms:
        # Creates a name for each join state argument that corresponds to the
        # variable name, but reformatting the state_var to avoid IR variable
        # characters like "$"
        state_name = "".join(
            char for char in state_var.name if char.isalnum() or char == "_"
        )
        rtjf_state_col_indices.append(col_indices)
        rtjf_state_args.append(state_var)
        rtjf_state_names.append(state_name)
        rtjf_non_equality_conditions.append(non_equality_conditions)
    return (
        rtjf_state_col_indices,
        rtjf_state_args,
        rtjf_state_names,
        rtjf_non_equality_conditions,
    )


def get_rtjf_cols_extra_info(column_types, desired_indices):
    """
    Fetches the column precisions and timezones for any columns where this information
    is required for min/max I/O runtime join filters.
    """
    precisions = []
    time_zones = []
    for col_idx in desired_indices:
        if isinstance(column_types[col_idx], bodo.types.TimeArrayType):
            precisions.append(column_types[col_idx].precision)
            time_zones.append(None)
        elif isinstance(column_types[col_idx], bodo.types.DatetimeArrayType):
            precisions.append(-1)
            time_zones.append(column_types[col_idx].tz)
        else:
            precisions.append(-1)
            time_zones.append(None)
    return precisions, time_zones


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
        op_id_msg = (
            f" (Operator ID: {sql_node.sql_op_id}) " if sql_node.sql_op_id != -1 else ""
        )
        pruning_msg = (
            "Finish column pruning on read_sql node%s:\n%s\nColumns loaded %s\n"
        )
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
            op_id_msg,
            sql_source,
            sql_cols,
        )
        # Log if any columns use dictionary encoded arrays.
        if dict_encoded_cols:
            encoding_msg = "Finished optimized encoding on read_sql node%s:\n%s\nColumns %s using dictionary encoding to reduce memory usage.\n"
            bodo.user_logging.log_message(
                "Dictionary Encoding",
                encoding_msg,
                op_id_msg,
                sql_source,
                dict_encoded_cols,
            )
        if bodo.user_logging.get_verbose_level() >= 2:
            io_msg = "read_sql %s table/query:\n%s\n\nColumns/Types:\n"
            for c, t in zip(sql_cols, sql_types):
                io_msg += f"{c}: {t}\n"
            bodo.user_logging.log_message(
                "SQL I/O",
                io_msg,
                op_id_msg,
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
                "pandas.read_sql(): 1 or more columns found with Arrow types that are not supported in Bodo and could not be eliminated. "
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
        limit = min(sql_node.limit, meta_head_only_info[0])

    # If we have runtime join filter terms passed in, add them to the query code.
    rtjf_suffix = ""
    rtjf_state_args = []
    rtjf_state_names = []
    if sql_node.rtjf_terms is not None:
        (
            rtjf_state_cols,
            rtjf_state_args,
            rtjf_state_names,
            rtjf_non_equality_info,
        ) = extract_rtjf_terms(sql_node.rtjf_terms)
        rtjf_suffix += '    runtime_join_filter_conds = ["TRUE"]\n'
        for i in range(len(rtjf_state_names)):
            state_var = rtjf_state_names[i]
            col_indices = rtjf_state_cols[i]
            non_equality_info = rtjf_non_equality_info[i]
            state_ir_var = sql_node.rtjf_terms[i][0]
            state_type = typemap[state_ir_var.name]
            build_indices = state_type.build_indices
            # Fetch the precision and time zone for each of the used columns
            if len(col_indices) > 0:
                precisions, time_zones = get_rtjf_cols_extra_info(
                    sql_node.out_table_col_types, col_indices
                )
                rtjf_suffix += f"    runtime_join_filter_conds.extend(gen_runtime_join_filter_cond({state_var}, {col_indices}, {precisions}, {time_zones}))\n"
            if len(non_equality_info) > 0:
                probe_cols = [x[0] for x in non_equality_info]
                # Remap build indices for C++
                build_cols = [build_indices[x[1]] for x in non_equality_info]
                ops = [x[2] for x in non_equality_info]
                precisions, time_zones = get_rtjf_cols_extra_info(
                    sql_node.out_table_col_types, probe_cols
                )
                rtjf_suffix += f"    runtime_join_filter_conds.extend(gen_runtime_join_filter_interval_cond({state_var}, {probe_cols}, {build_cols}, {ops}, {precisions}, {time_zones}))\n"
        rtjf_suffix += (
            '    runtime_join_filter_cond = " AND ".join(runtime_join_filter_conds)\n'
        )
        rtjf_suffix += '    sql_request = f"SELECT * FROM ({sql_request}) WHERE {runtime_join_filter_cond}"\n'

        if bodo.user_logging.get_verbose_level() >= 2:
            rtjf_suffix += "    log_message('SQL I/O', f'Runtime join filter query: {sql_request}')\n"

    filter_map, filter_vars = bodo.ir.connector.generate_filter_map(sql_node.filters)
    extra_args = ", ".join(list(filter_map.values()) + rtjf_state_names)
    func_text = f"def sql_impl(sql_request, conn, database_schema, {extra_args}):\n"
    # If we are doing regular SQL, filters are embedded into the query.
    if sql_node.is_select_query:
        if sql_node.filters:  # pragma: no cover
            visitor = SnowflakeFilterVisitor(
                filter_map, sql_node.converted_colnames, typemap
            )
            where_cond = " WHERE " + visitor.visit(sql_node.filters)
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
        func_text += rtjf_suffix

    filter_args = ""

    # total_rows is used for setting total size variable below
    if sql_node.is_streaming:  # pragma: no cover
        func_text += f"    snowflake_reader = _sql_reader_py(sql_request, conn, database_schema, {filter_args})\n"
    else:
        func_text += f"    (total_rows, table_var, index_var) = _sql_reader_py(sql_request, conn, database_schema, {filter_args})\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    sql_impl = loc_vars["sql_impl"]

    genargs: dict[str, Any] = {
        "col_names": sql_node.out_table_col_names,
        "col_typs": sql_node.out_table_col_types,
        "index_column_name": sql_node.index_column_name,
        "index_column_type": sql_node.index_column_type,
        "out_used_cols": sql_node.out_used_cols,
        "converted_colnames": sql_node.converted_colnames,
        "db_type": sql_node.db_type,
        "limit": limit,
        "parallel": parallel,
        "is_dead_table": not sql_node.is_live_table,
        "is_select_query": sql_node.is_select_query,
        "is_independent": is_independent,
        "downcast_decimal_to_double": sql_node.downcast_decimal_to_double,
        "pyarrow_schema": sql_node.pyarrow_schema,
    }
    if sql_node.is_streaming:
        assert sql_node.chunksize is not None
        sql_reader_py = _gen_snowflake_reader_chunked_py(
            **genargs,
            chunksize=sql_node.chunksize,
            sql_op_id=sql_node.sql_op_id,
        )
    else:
        sql_reader_py = _gen_sql_reader_py(**genargs)

    schema_type = types.none if sql_node.database_schema is None else string_type
    f_block = compile_to_numba_ir(
        sql_impl,
        {
            "_sql_reader_py": sql_reader_py,
            "bcast_scalar": bcast_scalar,
            "get_sql_literal": _get_snowflake_sql_literal,
            "gen_runtime_join_filter_cond": gen_runtime_join_filter_cond,
            "gen_runtime_join_filter_interval_cond": gen_runtime_join_filter_interval_cond,
            "log_message": bodo.user_logging.log_message,
        },
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=(string_type, string_type, schema_type)
        + tuple(typemap[v.name] for v in filter_vars)
        + tuple(typemap[v.name] for v in rtjf_state_args),
        typemap=typemap,
        calltypes=calltypes,
    ).blocks.popitem()[1]

    if sql_node.is_select_query:
        # Prune the columns to only those that are used.
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
        + filter_vars
        + rtjf_state_args,
    )
    nodes = f_block.body[:-3]

    # Set total size variable if necessary (for limit pushdown, iceberg specific)
    # value comes from 'total_rows' output of '_sql_reader_py' above
    if meta_head_only_info:
        nodes[-3].target = meta_head_only_info[1]

    if sql_node.is_streaming:  # pragma: no cover
        nodes[-1].target = sql_node.out_vars[0]
        return nodes

    # assign output table
    nodes[-2].target = sql_node.out_vars[0]
    # assign output index array
    nodes[-1].target = sql_node.out_vars[1]
    # At most one of the table and the index
    # can be dead because otherwise the whole
    # node should have already been removed.
    assert sql_node.has_side_effects or not (
        sql_node.index_column_name is None and not sql_node.is_live_table
    ), (
        "At most one of table and index should be dead if the SQL IR node is live and has no side effects"
    )
    if sql_node.index_column_name is None:
        # If the index_col is dead, remove the node.
        nodes.pop(-1)
    elif not sql_node.is_live_table:
        # If the table is dead, remove the node
        nodes.pop(-2)

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
    elif isinstance(filter_type, bodo.types.PandasTimestampType):
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
    elif filter_type == bodo.types.datetime_date_type:
        # datetime.date needs to be converted to a date literal
        # Just return the string wrapped in quotes.
        # https://docs.snowflake.com/en/sql-reference/data-types-datetime.html#date
        return (
            lambda filter_value: f"date '{filter_value.strftime('%Y-%m-%d')}'"
        )  # pragma: no cover
    elif filter_type == bodo.types.datetime64ns:
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
    scalar_isinstance = (types.Integer, types.Float, bodo.types.PandasTimestampType)
    scalar_equals = (
        bodo.types.datetime_date_type,
        types.unicode_type,
        types.bool_,
        bodo.types.datetime64ns,
        types.none,
    )
    filter_type = types.unliteral(filter_value)
    if (
        isinstance(
            filter_type,
            (
                types.List,
                types.Array,
                bodo.types.IntegerArrayType,
                bodo.types.FloatingArrayType,
                bodo.types.DatetimeArrayType,
            ),
        )
        or filter_type
        in (
            bodo.types.string_array_type,
            bodo.types.dict_str_arr_type,
            bodo.types.boolean_array_type,
            bodo.types.datetime_date_array_type,
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
        # Snowflake doesn't require reading any columns
        require_one_column=sql_node.db_type != "snowflake",
    )


numba.parfors.array_analysis.array_analysis_extensions[SqlReader] = (
    bodo.ir.connector.connector_array_analysis
)
distributed_analysis.distributed_analysis_extensions[SqlReader] = (
    bodo.ir.connector.connector_distributed_analysis
)
typeinfer.typeinfer_extensions[SqlReader] = bodo.ir.connector.connector_typeinfer
ir_utils.visit_vars_extensions[SqlReader] = bodo.ir.connector.visit_vars_connector
ir_utils.remove_dead_extensions[SqlReader] = remove_dead_sql
numba.core.analysis.ir_extension_usedefs[SqlReader] = (
    bodo.ir.connector.connector_usedefs
)
ir_utils.copy_propagate_extensions[SqlReader] = bodo.ir.connector.get_copies_connector
ir_utils.apply_copy_propagate_extensions[SqlReader] = (
    bodo.ir.connector.apply_copies_connector
)
ir_utils.build_defs_extensions[SqlReader] = (
    bodo.ir.connector.build_connector_definitions
)
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
    with bodo.ir.object_mode.no_warning_objmode():
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
    with bodo.ir.object_mode.no_warning_objmode():
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
    with bodo.ir.object_mode.no_warning_objmode():
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
    with bodo.ir.object_mode.no_warning_objmode():
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
    index_column_name: str | None,
    index_column_type: types.ArrayCompatible | types.NoneType,
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
    index_column_name: str | None,
    index_column_type,
    out_used_cols: list[int],
    converted_colnames: list[str],
    db_type: str,
    limit: int | None,
    parallel: bool,
    pyarrow_schema: pa.Schema | None,
    is_dead_table: bool,
    is_select_query: bool,
    is_independent: bool,
    downcast_decimal_to_double: bool,
    chunksize: int,
    sql_op_id: int,
):  # pragma: no cover
    """Function to generate main streaming SQL implementation.

    See _gen_sql_reader_py for argument documentation

    Args:
        chunksize: Number of rows in each batch
    """
    assert db_type == "snowflake", (
        f"Database {db_type} not supported in streaming IO mode, and should not go down this path"
    )
    assert pyarrow_schema is not None, (
        "SQLNode must contain a pyarrow_schema if reading from Snowflake"
    )

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
            "  total_rows_np = np.array([0], dtype=np.int64)",
            "  snowflake_reader = snowflake_reader_init_py_entry(",
            "    unicode_to_utf8(sql_request),",
            "    unicode_to_utf8(conn),",
            f"    {parallel},",
            f"    {is_independent},",
            f"    pyarrow_schema_{call_id},",
            f"    {len(params.nullable_cols_array)},",
            "    nullable_cols_array.ctypes,",
            f"    {len(params.snowflake_dict_cols_array)},",
            "    snowflake_dict_cols_array.ctypes,",
            "    total_rows_np.ctypes,",
            f"    {is_select_query and len(used_col_names) == 0},",
            f"    {is_select_query},",
            f"    {downcast_decimal_to_double},",
            f"    {chunksize},",
            f"    {sql_op_id},",
            "    out_type,",
            "  )",
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


def _gen_sql_reader_py(
    col_names: list[str],
    col_typs: list[Any],
    index_column_name: str | None,
    index_column_type,
    out_used_cols: list[int],
    converted_colnames: list[str],
    db_type: str,
    limit: int | None,
    parallel: bool,
    pyarrow_schema: pa.Schema | None,
    is_dead_table: bool,
    is_select_query: bool,
    is_independent: bool,
    downcast_decimal_to_double: bool,
):
    """
    Function that generates the main SQL implementation. There are
    three main implementation paths:
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
        pyarrow_schema: PyArrow schema for the source table. This should only
            be used for Snowflake.
        is_select_query: Are we executing a select?
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
    func_text = "def sql_reader_py(sql_request, conn, database_schema):\n"

    if db_type == "snowflake":  # pragma: no cover
        assert pyarrow_schema is not None, (
            "SQLNode must contain a pyarrow_schema if reading from Snowflake"
        )

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
        func_text += "  total_rows = total_rows_np[0]\n"
        if parallel:
            func_text += "  local_rows = get_node_portion(total_rows, bodo.get_size(), bodo.get_rank())\n"
        else:
            func_text += "  local_rows = total_rows\n"
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
                        "  table_var = set_table_len(table_var, len(index_var))\n"
                    )
                else:
                    # Set the table length using the total rows if don't load any columns
                    func_text += "  table_var = set_table_len(table_var, local_rows)\n"
        func_text += "  delete_table(out_table)\n"
        func_text += "  ev.finalize()\n"
        func_text += "  return (total_rows, table_var, index_var)\n"

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
                func_text += (
                    '  with bodo.ir.object_mode.no_warning_objmode(nb_row="int64"):\n'
                )
                func_text += f"     if rank == {DEFAULT_ROOT}:\n"
                func_text += "         sql_cons = 'select count(*) from (' + sql_request + ') x'\n"
                func_text += "         frame = pd.read_sql(sql_cons, conn)\n"
                func_text += "         nb_row = frame.iat[0,0]\n"
                func_text += "     else:\n"
                func_text += "         nb_row = 0\n"
                func_text += "  nb_row = bcast_scalar(nb_row)\n"
            func_text += f"  with bodo.ir.object_mode.no_warning_objmode(table_var=py_table_type_{call_id}, index_var=index_col_typ):\n"
            func_text += "    offset, limit = bodo.libs.distributed_api.get_start_count(nb_row)\n"
            # https://docs.oracle.com/javadb/10.8.3.0/ref/rrefsqljoffsetfetch.html
            if db_type == "oracle":
                func_text += "    sql_cons = 'select * from (' + sql_request + ') OFFSET ' + str(offset) + ' ROWS FETCH NEXT ' + str(limit) + ' ROWS ONLY'\n"
            else:
                func_text += "    sql_cons = 'select * from (' + sql_request + ') x LIMIT ' + str(limit) + ' OFFSET ' + str(offset)\n"

            func_text += "    df_ret = pd.read_sql(sql_cons, conn)\n"
            func_text += (
                "    bodo.ir.connector.cast_float_to_nullable(df_ret, df_typeref_2)\n"
            )
        else:
            func_text += f"  with bodo.ir.object_mode.no_warning_objmode(table_var=py_table_type_{call_id}, index_var=index_col_typ):\n"
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
            func_text += "    arrs = []\n"
            func_text += "    for i in range(df_ret.shape[1]):\n"
            func_text += "      arrs.append(df_ret.iloc[:, i].values)\n"
            # Bodo preserves all of the original types needed at typing in col_typs
            func_text += f"    table_var = Table(arrs, type_usecols_offsets_arr_{call_id}_2, {len(col_names)})\n"
        else:
            # Dead Table
            func_text += "    table_var = None\n"
        func_text += "  return (-1, table_var, index_var)\n"

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
                f"table_idx_{call_id}": table_idx,
                f"pyarrow_schema_{call_id}": pyarrow_schema,
                "unicode_to_utf8": unicode_to_utf8,
                "check_and_propagate_cpp_exception": check_and_propagate_cpp_exception,
                "array_from_cpp_table": array_from_cpp_table,
                "delete_table": delete_table,
                "cpp_table_to_py_table": cpp_table_to_py_table,
                "set_table_len": bodo.hiframes.table.set_table_len,
                "get_node_portion": bodo.libs.distributed_api.get_node_portion,
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
                "bcast_scalar": bcast_scalar,
                "pymysql_check": pymysql_check,
                "cx_oracle_check": cx_oracle_check,
                "psycopg2_check": psycopg2_check,
                "df_typeref": bodo.types.DataFrameType(
                    tuple(used_col_types),
                    bodo.types.RangeIndexType(None),
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


@intrinsic
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
    op_id_t,
    arrow_reader_t,
):  # pragma: no cover
    assert isinstance(arrow_reader_t, types.TypeRef) and isinstance(
        arrow_reader_t.instance_type, ArrowReaderType
    ), (
        "snowflake_reader_init_py_entry(): The last argument arrow_reader must by a TypeRef to an ArrowReader"
    )
    assert pyarrow_schema_t == pyarrow_schema_type, (
        "snowflake_reader_init_py_entry(): The 5th argument pyarrow_schema must by a PyArrow schema"
    )

    def codegen(context: BaseContext, builder: IRBuilder, signature, args):
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
                lir.IntType(64),  # op_id
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
        types.int64,  # op_id
        arrow_reader_t,  # typing only
    )
    return sig, codegen
