from __future__ import annotations

import datetime
import typing as pt

import llvmlite.binding as ll
import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
from llvmlite import ir as lir
from numba.core import cgutils, ir, ir_utils, typeinfer, types
from numba.core.extending import overload
from numba.core.ir_utils import (
    compile_to_numba_ir,
    next_label,
    replace_arg_nodes,
)
from numba.extending import intrinsic

import bodo
import bodo.ir.connector
import bodo.ir.filter as bif
import bodo.user_logging
from bodo.hiframes.table import TableType
from bodo.io import arrow_cpp  # type: ignore
from bodo.io.arrow_reader import ArrowReaderType
from bodo.io.helpers import pyarrow_schema_type
from bodo.io.iceberg.common import IcebergConnectionType
from bodo.io.parquet_pio import ParquetFilterScalarsListType, ParquetPredicateType
from bodo.ir.connector import Connector, log_limit_pushdown
from bodo.ir.filter import Filter, FilterVisitor
from bodo.ir.sql_ext import (
    RtjfValueType,
    extract_rtjf_terms,
    get_rtjf_cols_extra_info,
    get_runtime_join_filter_min_max,
    get_runtime_join_filter_unique_values,
    has_runtime_join_filter_unique_values,
    rtjf_term_repr,
)
from bodo.libs.array import (
    array_from_cpp_table,
    cpp_table_to_py_table,
    delete_table,
    table_type,
)
from bodo.libs.distributed_api import bcast_scalar
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
    is_nullable_ignore_sentinels,
    raise_bodo_error,
)
from bodo.utils.utils import (
    check_and_propagate_cpp_exception,
    inlined_check_and_propagate_cpp_exception,
)

if pt.TYPE_CHECKING:  # pragma: no cover
    from llvmlite.ir.builder import IRBuilder
    from numba.core.base import BaseContext
    from pyiceberg.expressions.literals import Literal


ll.add_symbol("iceberg_pq_read_py_entry", arrow_cpp.iceberg_pq_read_py_entry)
ll.add_symbol(
    "iceberg_pq_reader_init_py_entry", arrow_cpp.iceberg_pq_reader_init_py_entry
)


parquet_predicate_type = ParquetPredicateType()
parquet_filter_scalars_list_type = ParquetFilterScalarsListType()


@intrinsic
def iceberg_pq_read_py_entry(
    typingctx,
    conn_str,
    table_id,
    parallel,
    limit,
    dnf_filters,
    expr_filter_f_str,
    filter_scalars,
    selected_cols,
    num_selected_cols,
    nullable_cols,
    pyarrow_schema,
    dict_encoded_cols,
    num_dict_encoded_cols,
    create_dict_from_string,
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
        sql_request_str (types.voidptr): C string for the Iceberg table identifier
        parallel (types.boolean): Is the read in parallel
        limit (types.int64): Max number of rows to read. -1 if all rows
        dnf_filters (parquet_predicate_type): PyObject for DNF filters.
        expr_filter_f_str (types.voidptr): f-string representation of the
            expression filter. This is used to generate the filter expression
            at runtime.
        filter_scalars (parquet_filter_scalars_list_type): Scalars to use
            to generate the filter expression at runtime.
        selected_cols (types.voidptr): C pointers of integers for selected columns
        num_selected_cols (types.int64): Length of selected_cols
        nullable_cols (types.voidptr): C pointers of 0 or 1 for if each selected column is nullable
        pyarrow_schema (pyarrow_schema_type): Pyobject with the pyarrow schema for the output.
        dict_encoded_cols (types.voidptr): Array fo column numbers that are dictionary encoded.
        num_dict_encoded_cols (_type_): Length of dict_encoded_cols
        create_dict_from_string (bool): Whether the dict-encoding should be done in Bodo instead
            of Arrow.
        is_merge_into_cow (bool): Are we doing a merge?
    """

    def codegen(context, builder, signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(1),  # bool
                lir.IntType(64),  # int64
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(32),  # int32
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(8).as_pointer(),  # void*
                lir.IntType(32),  # int32
                lir.IntType(1),  # bool
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
        types.boolean,
        types.int64,
        parquet_predicate_type,  # dnf filters
        types.voidptr,  # expr_filter_f_str
        parquet_filter_scalars_list_type,  # filter_scalars
        types.voidptr,
        types.int32,
        types.voidptr,
        pyarrow_schema_type,
        types.voidptr,
        types.int32,
        types.boolean,
        types.boolean,
    )
    return sig, codegen


@intrinsic
def iceberg_pq_reader_init_py_entry(
    typingctx,
    conn_str,
    table_id,
    parallel,
    limit,
    dnf_filters,
    expr_filter_f_str,
    filter_scalars,
    selected_cols,
    num_selected_cols,
    nullable_cols,
    pyarrow_schema,
    dict_encoded_cols,
    num_dict_encoded_cols,
    create_dict_from_string,
    chunksize_t,
    op_id_t,
    arrow_reader_t,
):
    """Construct a reader for an Iceberg Table using a the C++
    iceberg_pq_reader_init_py_entry function. That function returns an ArrowReader

    Args:
        typingctx (Context): Context used for typing
        conn_str (types.voidptr): C string for the connection
        table_id (types.voidptr): C string for Iceberg table id
        parallel (types.boolean): Is the read in parallel
        limit (types.int64): Max number of rows to read. -1 if all rows
        dnf_filters (parquet_predicate_type): PyObject for DNF filters.
        expr_filter_f_str (types.voidptr): f-string representation of the
            expression filter. This is used to generate the filter expression
            at runtime.
        filter_scalars (parquet_filter_scalars_list_type): Scalars to use
            to generate the filter expression at runtime.
        selected_cols (types.voidptr): C pointers of integers for selected columns
        num_selected_cols (types.int64): Length of selected_cols
        nullable_cols (types.voidptr): C pointers of 0 or 1 for if each selected column is nullable
        pyarrow_schema (pyarrow_schema_type): Pyobject with the pyarrow schema for the output.
        dict_encoded_cols (types.voidptr): Array fo column numbers that are dictionary encoded.
        num_dict_encoded_cols (_type_): Length of dict_encoded_cols
        create_dict_from_string (bool): Whether the dict-encoding should be done by Bodo (i.e. read
            as strings from Arrow and then dict-encode them) instead of Arrow (i.e. read as dict-encoded
            string arrays directly).
        chunksize: Chunk size to use for streaming read.
        op_id: Operator ID generated by planner for query profile purposes.
        arrow_reader_t (ArrowReader): The typing of the output ArrowReader
    """

    assert isinstance(arrow_reader_t, types.TypeRef) and isinstance(
        arrow_reader_t.instance_type, ArrowReaderType
    ), (
        "iceberg_pq_reader_init_py_entry(): The last argument arrow_reader must by a TypeRef to an ArrowReader"
    )

    def codegen(context: BaseContext, builder: IRBuilder, signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),  # conn_str void*
                lir.IntType(8).as_pointer(),  # table_id void*
                lir.IntType(1),  # parallel bool
                lir.IntType(64),  # tot_rows_to_read int64
                lir.IntType(8).as_pointer(),  # dnf_filters void*
                lir.IntType(8).as_pointer(),  # expr_filter_f_str void*
                lir.IntType(8).as_pointer(),  # filter_scalars void*
                lir.IntType(8).as_pointer(),  # _selected_fields void*
                lir.IntType(32),  # num_selected_fields int32
                lir.IntType(8).as_pointer(),  # _is_nullable void*
                lir.IntType(8).as_pointer(),  # pyarrow_schema PyObject*
                lir.IntType(8).as_pointer(),  # _str_as_dict_cols void*
                lir.IntType(32),  # num_str_as_dict_cols int32
                lir.IntType(1),  # create_dict_from_string bool
                lir.IntType(64),  # chunksize int64
                lir.IntType(64),  # op_id int64
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
        types.boolean,
        types.int64,
        parquet_predicate_type,  # dnf filters
        types.voidptr,  # expr_filter_f_str
        parquet_filter_scalars_list_type,  # filter_scalars
        types.voidptr,
        types.int32,
        types.voidptr,
        pyarrow_schema_type,  # pyarrow_schema
        types.voidptr,  # str_as_dict_cols
        types.int32,  # num_str_as_dict_cols
        types.boolean,  # create_dict_from_string
        types.int64,  # chunksize
        types.int64,  # op_id
        arrow_reader_t,  # typing only
    )
    return sig, codegen


class IcebergReader(Connector):
    connector_typ: str = "iceberg"

    def __init__(
        self,
        table_id: str,
        connection: ir.AbstractRHS,
        df_out_varname: str,
        out_table_col_names: list[str],
        out_table_col_types: list[types.ArrayCompatible],
        out_vars: list[ir.Var],
        loc: ir.Loc,
        unsupported_columns: list[str],
        unsupported_arrow_types: list[pa.DataType],
        index_column_name: str | None,
        index_column_type: types.ArrayCompatible | types.NoneType,
        pyarrow_schema: pa.Schema,
        # Only relevant for Iceberg MERGE INTO COW
        is_merge_into: bool,
        file_list_type: types.Type,
        snapshot_id_type: types.Type,
        # Batch size to read chunks in, or none, to read the entire table together
        # Only supported for Snowflake
        # Treated as compile-time constant for simplicity
        # But not enforced that all chunks are this size
        chunksize: int | None = None,
        used_cols: list[str] | None = None,
        initial_filter: Filter | None = None,
        initial_limit: int | None = None,
        orig_col_names=None,
        orig_col_types=None,
        # Operator ID generated by BodoSQL for query profile
        # purposes. Only supported in the streaming case.
        sql_op_id: int = -1,
        # Whether we should do dictionary-encoding in Bodo after
        # reading the columns as strings from Arrow (similar to
        # the Snowflake Reader), or should we let Arrow return
        # dict-encoded columns directly.
        dict_encode_in_bodo: bool = False,
        # List of tuples representing runtime join filters
        # that have been pushed down to I/O.
        rtjf_terms: list[tuple[ir.Var, tuple[int], tuple[int, int, str]]] | None = None,
    ):
        # Info required to connect to the catalog and table
        self.table_id = table_id
        self.connection = connection

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

        self.df_out_varname = df_out_varname  # used only for printing
        self.out_vars = out_vars
        self.loc = loc
        # Support for filter pushdown
        self.filters = initial_filter

        # List of indices within the table name that are used.
        # out_table_col_names is unchanged unless the table is deleted,
        # so this is used to track dead columns.
        self.out_used_cols = list(range(len(out_table_col_names)))
        # This is the PyArrow schema object.
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

        self.chunksize = chunksize
        self.used_cols = used_cols
        self.orig_col_names = orig_col_names
        self.orig_col_types = orig_col_types

        self.limit = initial_limit
        # Log limit pushdown from BodoSQL
        if self.limit is not None:
            log_limit_pushdown(self, self.limit)

        self.sql_op_id = sql_op_id

        # Whether we should do dictionary-encoding in Bodo after
        # reading the columns as strings from Arrow (similar to
        # the Snowflake Reader), or should we let Arrow return
        # dict-encoded columns directly. This is set to True
        # in the case of Snowflake managed Iceberg tables since
        # they sometimes write strings with encodings that Arrow
        # cannot read as dict-encoded columns directly.
        self.dict_encode_in_bodo = dict_encode_in_bodo

        self.rtjf_terms = rtjf_terms

    def __repr__(self) -> str:  # pragma: no cover
        out_varnames = tuple(v.name for v in self.out_vars)
        runtime_join_filters = rtjf_term_repr(self.rtjf_terms)
        return (
            f"{out_varnames} = IcebergReader({self.table_id=}, {self.connection=}, "
            f"{self.out_table_col_names=}, {self.out_table_col_types=}, "
            f"{self.df_out_varname=}, {self.unsupported_columns=}, "
            f"{self.unsupported_arrow_types=}, {self.index_column_name=}, "
            f"{self.index_column_type=}, {self.out_used_cols=}, "
            f"{self.pyarrow_schema=}, {self.is_merge_into=}, "
            f"{self.sql_op_id=}, {self.dict_encode_in_bodo=}, {runtime_join_filters=})"
        )

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
        # Output is not guaranteed to be equally distributed
        # since there might be differences in the sizes of files.
        return Distribution.OneD_Var


def remove_dead_iceberg(
    iceberg_node: IcebergReader,
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
    if iceberg_node.is_streaming:  # pragma: no cover
        return iceberg_node

    table_var = iceberg_node.out_vars[0].name
    index_var = iceberg_node.out_vars[1].name
    file_list_var = (
        iceberg_node.out_vars[2].name if len(iceberg_node.out_vars) > 2 else None
    )
    snapshot_id_var = (
        iceberg_node.out_vars[3].name if len(iceberg_node.out_vars) > 3 else None
    )
    if (
        table_var not in lives
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
        iceberg_node.out_table_col_names = []
        iceberg_node.out_table_col_types = []
        iceberg_node.out_used_cols = []
        iceberg_node.is_live_table = False

    if index_var not in lives:
        # If the index_var not in lives we don't load the index.
        # To do this we mark the index_column_name as None
        iceberg_node.index_column_name = None
        iceberg_node.index_column_type = types.none

    if file_list_var not in lives:
        iceberg_node.file_list_live = False
        iceberg_node.file_list_type = types.none

    if snapshot_id_var not in lives:
        iceberg_node.snapshot_id_live = False
        iceberg_node.snapshot_id_type = types.none
    return iceberg_node


def iceberg_remove_dead_column(
    iceberg_node: IcebergReader, column_live_map, equiv_vars, typemap
):
    """
    Function that tracks which columns to prune from the SQL node.
    This updates out_used_cols which stores which arrays in the
    types will need to actually be loaded.

    This is mapped to the used column names during distributed pass.
    """
    return bodo.ir.connector.base_connector_remove_dead_columns(
        iceberg_node,
        column_live_map,
        equiv_vars,
        typemap,
        "IcebergReader",
        # out_table_col_names is set to an empty list if the table is dead
        # see 'remove_dead_sql'
        iceberg_node.out_table_col_names,
        # Iceberg and Snowflake don't require reading any columns
        require_one_column=False,
    )


# XXX: temporary fix pending Numba's #3378
# keep the compiled functions around to make sure GC doesn't delete them and
# the reference to the dynamic function inside them
# (numba/lowering.py:self.context.add_dynamic_addr ...)
compiled_funcs = []


def iceberg_distributed_run(
    iceberg_node: IcebergReader,
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
            f" (Operator ID: {iceberg_node.sql_op_id}) "
            if iceberg_node.sql_op_id != -1
            else ""
        )
        pruning_msg = (
            "Finish column pruning on read_sql node%s:\n%s\nColumns loaded %s\n"
        )
        sql_cols = []
        sql_types = []
        dict_encoded_cols = []
        out_types = iceberg_node.out_table_col_types
        for i in iceberg_node.out_used_cols:
            colname = iceberg_node.out_table_col_names[i]
            sql_cols.append(colname)
            sql_types.append(out_types[i])
            if isinstance(out_types[i], bodo.libs.dict_arr_ext.DictionaryArrayType):
                dict_encoded_cols.append(colname)
        # Include the index since it needs to be loaded from the query
        if iceberg_node.index_column_name:
            sql_cols.append(iceberg_node.index_column_name)
            if isinstance(
                iceberg_node.index_column_type,
                bodo.libs.dict_arr_ext.DictionaryArrayType,
            ):
                dict_encoded_cols.append(iceberg_node.index_column_name)
        sql_source = iceberg_node.loc.strformat()
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
                iceberg_node.table_id,
            )

    if iceberg_node.is_streaming:  # pragma: no cover
        parallel = bodo.ir.connector.is_chunked_connector_table_parallel(
            iceberg_node, array_dists, "SQLReader"
        )
    else:
        parallel = bodo.ir.connector.is_connector_table_parallel(
            iceberg_node, array_dists, typemap, "SQLReader"
        )

    # Check for any unsupported columns still remaining
    if iceberg_node.unsupported_columns:
        # Determine the columns that were eliminated.
        unsupported_cols_set = set(iceberg_node.unsupported_columns)
        used_cols_set = set(iceberg_node.out_used_cols)
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
                while iceberg_node.unsupported_columns[idx] != col_num:
                    idx += 1
                msg_list.append(
                    f"Column '{iceberg_node.unsupported_columns[col_num]}' with unsupported arrow type {iceberg_node.unsupported_arrow_types[idx]}"
                )
                idx += 1
            total_msg = "\n".join(msg_list)
            raise BodoError(total_msg, loc=iceberg_node.loc)

    # Generate the limit
    if iceberg_node.limit is None and (
        not meta_head_only_info or meta_head_only_info[0] is None
    ):
        # There is no limit
        limit = None
    elif iceberg_node.limit is None and meta_head_only_info is not None:
        # There is only limit pushdown
        limit = meta_head_only_info[0]
    elif not meta_head_only_info or meta_head_only_info[0] is None:
        # There is only a limit already in the query
        limit = iceberg_node.limit
    else:
        assert iceberg_node.limit is not None and meta_head_only_info[0] is not None
        # There is limit pushdown and a limit already in the query.
        # Compute the min to minimize compute.
        limit = min(iceberg_node.limit, meta_head_only_info[0])

    filter_map, filter_vars = bodo.ir.connector.generate_filter_map(
        iceberg_node.filters
    )

    if iceberg_node.rtjf_terms is not None:
        (
            rtjf_cols,
            rtjf_states_vars,
            rtjf_states_vars_names,
            rtjf_non_equality_info,
        ) = extract_rtjf_terms(iceberg_node.rtjf_terms)
        rtjf_build_indices_list = []
        for rtjf_term in iceberg_node.rtjf_terms:
            state_ir_var = rtjf_term[0]
            state_type = typemap[state_ir_var.name]
            build_indices = state_type.build_indices
            rtjf_build_indices_list.append(build_indices)
    else:
        (
            rtjf_cols,
            rtjf_states_vars,
            rtjf_states_vars_names,
            rtjf_non_equality_info,
            rtjf_build_indices_list,
        ) = (
            [],
            [],
            [],
            [],
            [],
        )

    extra_args = ", ".join(list(filter_map.values()) + list(rtjf_states_vars_names))
    func_text = f"def sql_impl(table_id, conn_wrapper, {extra_args}):\n"
    if isinstance(iceberg_node.connection, ir.Var) and isinstance(
        typemap[iceberg_node.connection.name], IcebergConnectionType
    ):
        func_text += "    conn = conn_wrapper.conn_str\n"
        conn_type = typemap[iceberg_node.connection.name]
    else:
        func_text += "    conn = conn_wrapper\n"
        conn_type = string_type

    filter_args = ""
    # Pass args to _iceberg_reader_py with iceberg
    filter_args = ", ".join(filter_map.values())

    # total_rows is used for setting total size variable below
    if iceberg_node.is_streaming:  # pragma: no cover
        func_text += f"    reader = _iceberg_reader_py(table_id, conn, {extra_args})\n"
    else:
        func_text += f"    (total_rows, table_var, index_var, file_list, snapshot_id) = _iceberg_reader_py(table_id, conn, {filter_args})\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    sql_impl = loc_vars["sql_impl"]

    genargs: dict[str, pt.Any] = {
        "col_names": iceberg_node.out_table_col_names,
        "col_typs": iceberg_node.out_table_col_types,
        "index_column_name": iceberg_node.index_column_name,
        "index_column_type": iceberg_node.index_column_type,
        "out_used_cols": iceberg_node.out_used_cols,
        "limit": limit,
        "parallel": parallel,
        "typemap": typemap,
        "filters": iceberg_node.filters,
        "is_dead_table": not iceberg_node.is_live_table,
        "is_merge_into": iceberg_node.is_merge_into,
        "pyarrow_schema": iceberg_node.pyarrow_schema.remove_metadata(),
        "dict_encode_in_bodo": iceberg_node.dict_encode_in_bodo,
    }
    if iceberg_node.is_streaming:
        assert iceberg_node.chunksize is not None
        sql_reader_py = _gen_iceberg_reader_chunked_py(
            **genargs,
            chunksize=iceberg_node.chunksize,
            used_cols=iceberg_node.used_cols,
            orig_col_names=iceberg_node.orig_col_names,
            orig_col_types=iceberg_node.orig_col_types,
            sql_op_id=iceberg_node.sql_op_id,
            rtjf_states_vars_names=rtjf_states_vars_names,
            rtjf_cols=[np.array(cols) for cols in rtjf_cols],
            rtjf_interval_cols=rtjf_non_equality_info,
            rtjf_build_indices_list=rtjf_build_indices_list,
        )
    else:
        sql_reader_py = _gen_iceberg_reader_py(**genargs)

    f_block = compile_to_numba_ir(
        sql_impl,
        {
            "_iceberg_reader_py": sql_reader_py,
            "bcast_scalar": bcast_scalar,
        },
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=(string_type, conn_type)
        + tuple(typemap[v.name] for v in filter_vars)
        + tuple(typemap[v.name] for v in rtjf_states_vars),
        typemap=typemap,
        calltypes=calltypes,
    ).blocks.popitem()[1]
    replace_arg_nodes(
        f_block,
        [
            ir.Const(iceberg_node.table_id, iceberg_node.loc),
            iceberg_node.connection,
        ]
        + filter_vars
        + rtjf_states_vars,
    )
    nodes = f_block.body[:-3]

    # Set total size variable if necessary (for limit pushdown, iceberg specific)
    # value comes from 'total_rows' output of '_iceberg_reader_py' above
    if meta_head_only_info:
        nodes[-5].target = meta_head_only_info[1]

    if iceberg_node.is_streaming:  # pragma: no cover
        nodes[-1].target = iceberg_node.out_vars[0]
        return nodes

    # assign output table
    nodes[-4].target = iceberg_node.out_vars[0]
    # assign output index array
    nodes[-3].target = iceberg_node.out_vars[1]
    # At most one of the table and the index
    # can be dead because otherwise the whole
    # node should have already been removed.
    assert not (
        iceberg_node.index_column_name is None and not iceberg_node.is_live_table
    ), "At most one of table and index should be dead if the Iceberg IR node is live"
    if iceberg_node.index_column_name is None:
        # If the index_col is dead, remove the node.
        nodes.pop(-3)
    elif not iceberg_node.is_live_table:
        # If the table is dead, remove the node
        nodes.pop(-4)

    # Do we load the file_list
    if iceberg_node.file_list_live:
        nodes[-2].target = iceberg_node.out_vars[2]
    else:
        nodes.pop(-2)
    # Do we load the snapshot_id
    if iceberg_node.snapshot_id_live:
        nodes[-1].target = iceberg_node.out_vars[3]
    else:
        nodes.pop(-1)

    return nodes


def get_filters_pyobject(filter_str, vars):  # pragma: no cover
    pass


try:
    import pyiceberg.expressions as pie
except ImportError:
    pie = None


def literal(val) -> Literal:
    """
    Wrapper over PyIceberg's literal function for constructing filters.
    This is needed to convert other Python literals to Iceberg-compatible ones
    """
    from pyiceberg.expressions.literals import DateLiteral, TimestampLiteral
    from pyiceberg.expressions.literals import literal as inner_literal
    from pyiceberg.utils.datetime import datetime_to_micros

    # PyIceberg literal doesn't support all data types
    # See https://github.com/apache/iceberg-python/issues/1456
    if isinstance(val, (datetime.datetime, pd.Timestamp)):
        return TimestampLiteral(datetime_to_micros(val))
    # pd.Timestamp and datetime.datetime are subclasses of datetime.date
    if isinstance(val, datetime.date):
        return DateLiteral((val - datetime.date(1970, 1, 1)).days)
    if isinstance(val, (list, pd.core.arrays.ExtensionArray)):
        return {literal(v) for v in val}  # type: ignore
    # TODO: Potentially need to support nested structures
    return inner_literal(val)


@overload(get_filters_pyobject, no_unliteral=True)
def overload_get_filters_pyobject(filters_str, var_tup):
    """generate a pyobject for filter expression to pass to C++"""
    if pie is None:
        raise_bodo_error("pyiceberg not found")

    filter_str_val = get_overload_const_str(filters_str)
    var_unpack = ", ".join(f"f{i}" for i in range(len(var_tup)))
    func_text = "def impl(filters_str, var_tup):\n"
    if len(var_tup):
        func_text += f"  {var_unpack}, = var_tup\n"
    func_text += (
        "  with bodo.no_warning_objmode(filters_py='parquet_predicate_type'):\n"
        f"    filters_py = {filter_str_val}\n"
        "  return filters_py\n"
    )

    loc_vars = {}
    glbs = globals().copy()
    glbs["bodo"] = bodo
    glbs["pie"] = pie
    glbs["literal"] = literal
    exec(func_text, glbs, loc_vars)

    return loc_vars["impl"]


class IcebergFilterVisitor(FilterVisitor[str]):
    """
    Convert a Bodo IR Filter term to a string representation
    of PyIceberg's filter classes.
    See filters_to_iceberg_expr for more details.

    Args:
        term: A Filter object representing a single filter term.
            A term is a tuple of the form (col_name, op, value).
            It can be a comparison or boolean operation, or a
            column conversion.
        filter_map: A dictionary mapping variable names to the
            corresponding filter scalar variables.

    Returns:
        A string representation of the FilterExpr object.
    """

    def __init__(self, filter_map):
        self.filter_map = filter_map

    def visit_scalar(self, scalar: bif.Scalar) -> str:
        return f"literal({self.filter_map[scalar.val.name]})"

    def visit_ref(self, ref: bif.Ref) -> str:
        return f"'{ref.val}'"

    def visit_op(self, op: bif.Op) -> str:
        op_name = op.op.upper()
        op_func = {
            ">": "GreaterThan",
            "<": "LessThan",
            ">=": "GreaterThanOrEqual",
            "<=": "LessThanOrEqual",
            "==": "EqualTo",
            "!=": "NotEqualTo",
            "STARTSWITH": "StartsWith",
            "IS_NULL": "IsNull",
            "IS_NOT_NULL": "NotNull",
            "IN": "In",
            "OR": "Or",
            "AND": "And",
            "NOT": "Not",
            "ALWAYS_TRUE": "AlwaysTrue",
            "ALWAYS_FALSE": "AlwaysFalse",
        }[op_name]
        return f"pie.{op_func}({', '.join(self.visit(x) for x in op.args)})"


def filters_to_iceberg_expr(filters: Filter | None, filter_map) -> str:
    """
    Convert a compiler Filter object to a string representation
    of PyIceberg's BooleanExpression
    Args:
        filters: A list of lists of Filter objects. Each inner list
            represents a disjunction of conjunctions of Filter objects.
        filter_map: A dictionary mapping variable names to the
            corresponding filter scalar variables.

    Returns:
        A string representation of the FilterExpr object.
    """

    if filters is None:
        return "pie.AlwaysTrue()"

    visitor = IcebergFilterVisitor(filter_map)
    dict_expr = visitor.visit(filters)

    if bodo.user_logging.get_verbose_level() >= 1:
        msg = "Iceberg Filter Pushed Down:\n%s\n"
        bodo.user_logging.log_message(
            "Filter Pushdown",
            msg,
            dict_expr,
        )

    return dict_expr


@numba.njit
def get_rtjf_col_min_max_unique_map(
    rtjf_state: bodo.libs.streaming.join.JoinStateType,
    rtjf_build_cols: npt.NDArray,
    rtjf_probe_cols: npt.NDArray,
    col_names: list[str],
    precisions: list[int],
    time_zones: list[str | None],
) -> tuple[
    list[str], list[tuple[RtjfValueType, RtjfValueType, RtjfValueType]]
]:  # pragma: no cover
    """
    Get the min/max bound for each key column in the runtime join filter supplied,
    or the list of unique values if it exists.
    """
    n_cols = len(rtjf_build_cols)
    out_col_names = []
    bounds = []
    for i in range(n_cols):
        build_col = rtjf_build_cols[i]
        probe_col = rtjf_probe_cols[i]
        tz = time_zones[i]
        # [BSE-3493] fix LTZ support with Iceberg
        if probe_col == -1 or tz is not None:
            continue
        # Get the min/max value bounds for the current key column from the join state
        if has_runtime_join_filter_unique_values(rtjf_state, build_col):
            unique_vals = get_runtime_join_filter_unique_values(rtjf_state, build_col)
        else:
            unique_vals = None
        min_val = get_runtime_join_filter_min_max(
            rtjf_state, build_col, True, precisions[i]
        )
        max_val = get_runtime_join_filter_min_max(
            rtjf_state, build_col, False, precisions[i]
        )
        col_name = col_names[probe_col]
        out_col_names.append(col_name)
        bounds.append((min_val, max_val, unique_vals))
    return out_col_names, bounds


@numba.njit
def add_rtjf_iceberg_filter(
    state_var,
    file_filters,
    filtered_cols: list[str],
    filter_ops: list[str],
    bounds: list[tuple[RtjfValueType, RtjfValueType, RtjfValueType]],
) -> ParquetPredicateType:  # pragma: no cover
    """
    For each column in filtered_cols create a FilterExpr containing it's bounds and combine them all with file_filters
    """
    assert pie is not None

    is_empty = bodo.ir.sql_ext.is_empty_build_table(state_var)
    with bodo.no_warning_objmode(combined_filters="parquet_predicate_type"):
        if is_empty:
            combined_filters = pie.AlwaysFalse()
        else:
            rtjf_filters = pie.AlwaysTrue()
            for col, (min, max, unique_vals), op in zip(
                filtered_cols, bounds, filter_ops
            ):
                if unique_vals is not None and len(unique_vals) > 0 and op == "==":
                    rtjf_filters = pie.And(
                        rtjf_filters, pie.In(col, literal(unique_vals))
                    )
                    print(rtjf_filters, unique_vals)
                else:
                    if min is not None and op in ("==", ">=", ">"):
                        rtjf_filters = pie.And(
                            rtjf_filters,
                            pie.GreaterThan(col, literal(min))
                            if op == ">"
                            else pie.GreaterThanOrEqual(col, literal(min)),
                        )
                    if max is not None and op in ("==", "<=", "<"):
                        rtjf_filters = pie.And(
                            rtjf_filters,
                            pie.LessThan(col, literal(max))
                            if op == "<"
                            else pie.LessThanOrEqual(col, literal(max)),
                        )

            combined_filters = pie.And(file_filters, rtjf_filters)

    return combined_filters


def convert_pyobj_to_arrow_filter_str(pyobj, tz):
    """
    Converts a Python object to the equivalent arrow
    representation that can be injected as a string into
    an arrow filter. For example:
    42 -> '42'
    "foo bar" -> "'foo bar'"
    datetime.date(2024, 1, 1) -> "pa.scalar(19723, pa.date32())"
        (since 2024-01-01 = 19723 days since 1970-01-01)
    bodo.Time(12, 30, 59, 0, 12, precision=6) -> "pa.scalar(45059000012, pa.time64('us'))"
    pd.Timestamp("2024-07-04 12:30:01.025601") -> "pa.scalar(1720096201025601000, pa.timestamp('ns'))"
    """
    if isinstance(pyobj, str):
        return f"'{pyobj}'"
    elif isinstance(pyobj, pd.Timestamp):
        if tz is not None:
            suffix = f", '{tz}'"
            # Convert the Timestamp from a UTC value to an equivalent value in the desired timezone,
            # thus allowing us to get the correct number of nanoseconds in the desired timezone.
            pyobj = pyobj.tz_localize("UTC").tz_convert(tz)
        else:
            suffix = ""
        return f"pa.scalar({pyobj.value}, pa.timestamp('ns'{suffix}))"
    elif isinstance(pyobj, datetime.date):
        since_1970 = pyobj.toordinal() - 719163
        return f"pa.scalar({since_1970}, pa.date32())"
    elif isinstance(pyobj, bodo.Time):
        if pyobj.precision == 0:
            return f"pa.scalar({pyobj.value}, pa.time64('s'))"
        elif pyobj.precision == 3:
            return f"pa.scalar({pyobj.value}, pa.time64('ms'))"
        elif pyobj.precision == 6:
            return f"pa.scalar({pyobj.value}, pa.time64('us'))"
        elif pyobj.precision == 9:
            return f"pa.scalar({pyobj.value}, pa.time64('ns'))"
        else:
            raise_bodo_error(f"Unsupported Time precision: {pyobj.precision}")
    else:
        return str(pyobj)


@numba.njit
def gen_runtime_join_filter_expr(
    filtered_cols: list[str],
    filter_ops: list[str],
    bounds: list[tuple[RtjfValueType, RtjfValueType, RtjfValueType]],
    time_zones: list[str | None],
) -> str:  # pragma: no cover
    """
    Create a string that evaluates to an arrow filter expression
    bounding each col to it's min/max as specified in rtjf_col_min_max_map
    Returns an f-string so column names can be substituted in at runtime based
    on each file's actual schema
    """
    rtjf_expr = ""

    with bodo.no_warning_objmode(rtjf_expr="unicode_type"):
        exprs = []
        for col, (min, max, unique_vals), op, tz in zip(
            filtered_cols, bounds, filter_ops, time_zones
        ):
            if unique_vals is not None and len(unique_vals) > 0 and op == "==":
                unique_strs = sorted(
                    [
                        convert_pyobj_to_arrow_filter_str(elem, tz)
                        for elem in unique_vals
                    ]
                )
                exprs.append(
                    f"(ds.field('{{{col}}}').isin([{', '.join(unique_strs)}]))"
                )
            else:
                if min is not None and op in ("==", ">=", ">"):
                    filter_op = ">=" if op == "==" else op
                    exprs.append(
                        f"(ds.field('{{{col}}}') {filter_op} {convert_pyobj_to_arrow_filter_str(min, tz)})"
                    )
                if max is not None and op in ("==", "<=", "<"):
                    filter_op = "<=" if op == "==" else op
                    exprs.append(
                        f"(ds.field('{{{col}}}') {filter_op} {convert_pyobj_to_arrow_filter_str(max, tz)})"
                    )
        rtjf_expr += " & ".join(exprs)
    return rtjf_expr


def _gen_iceberg_reader_chunked_py(
    col_names: list[str],
    col_typs: list[pt.Any],
    index_column_name: str | None,
    index_column_type,
    out_used_cols: list[int],
    limit: int | None,
    parallel: bool,
    typemap,
    filters: pt.Any | None,
    pyarrow_schema: pa.Schema | None,
    is_dead_table: bool,
    is_merge_into: bool,
    chunksize: int,
    used_cols: list[str] | None,
    orig_col_names,
    orig_col_types,
    dict_encode_in_bodo: bool,
    rtjf_states_vars_names: list[str],
    rtjf_cols: list[npt.NDArray[np.int32]],
    rtjf_interval_cols: list[tuple[tuple[int, int, str], ...]],
    rtjf_build_indices_list: list[tuple[int, ...]],
    sql_op_id: int = -1,
):  # pragma: no cover
    """Function to generate main streaming SQL implementation.

    See _gen_iceberg_reader_py for most argument documentation. A couple
    arguments that aren't forwarded are explained below.

    Args:
        chunksize: Number of rows in each batch
        rtjf_states_vars_names: Python variable name for each join state
        variable.
        rtjf_cols: List of each column used in equality rtjf for each join
        state.
        rtjf_interval_cols: List of each build column, probe column, and operator
        for each interval rtjf for each join state.
        rtjf_build_indices_list: List of build column indices for each join state.
        This is used for "remapping" the build columns in rtjf_interval_cols to the
        correct indices.
    """
    source_pyarrow_schema = pyarrow_schema
    assert source_pyarrow_schema is not None, (
        "SQLReader node must contain a source_pyarrow_schema if reading from Iceberg"
    )

    # Generate output pyarrow schema for used cols (from BodoSQL)
    if used_cols is None:  # pragma: no cover
        out_pyarrow_schema = source_pyarrow_schema
    else:  # pragma: no cover
        out_pyarrow_schema = pa.schema(
            [source_pyarrow_schema.field(i) for i in used_cols]
        )

    call_id = next_label()

    # Handle filter information because we may need to update the function header
    filter_args = ""
    filter_map = {}
    if filters:
        filter_map, _ = bodo.ir.connector.generate_filter_map(filters)
        filter_args = ", ".join(filter_map.values())

    # Generate the predicate filters. Note we pass
    # all col names as possible partitions via partition names.
    # The expression filters are returned as f-strings so that we can
    # pass them to the runtime to generate the filters dynamically
    # for the various schemas (to account for schema evolution).
    iceberg_expr_filter_f_str = bodo.ir.connector.generate_arrow_filters(
        filters,
        filter_map,
        orig_col_names,
        orig_col_names,
        orig_col_types,
        typemap,
        "iceberg",
        output_expr_filters_as_f_string=True,
        sql_op_id=sql_op_id,
    )
    filter_str = filters_to_iceberg_expr(filters, filter_map)

    # Determine selected C++ columns (and thus nullable) from original Iceberg
    # table / schema, assuming that Iceberg and Parquet field ordering is the same
    # Note that this does not include any locally generated columns (row id, file list, ...)
    out_selected_cols: list[int] = [
        out_pyarrow_schema.get_field_index(col_names[i]) for i in out_used_cols
    ]
    nullable_cols = [
        int(is_nullable_ignore_sentinels(col_typs[i])) for i in out_selected_cols
    ]
    source_selected_cols: list[int] = [
        source_pyarrow_schema.get_field_index(col_names[i]) for i in out_used_cols
    ]

    # pass indices to C++ of the selected string columns that are to be read
    # in dictionary-encoded format
    str_as_dict_cols = [
        src_idx
        for src_idx, out_idx in zip(source_selected_cols, out_selected_cols)
        if col_typs[out_idx] == bodo.dict_str_arr_type
    ]
    dict_str_cols_str = (
        f"dict_str_cols_arr_{call_id}.ctypes, np.int32({len(str_as_dict_cols)})"
        if str_as_dict_cols
        else "0, 0"
    )
    comma = "," if filter_args else ""
    rtjf_str = ""
    if len(rtjf_states_vars_names):
        rtjf_str = "  rtjf_exprs = []\n"
    probe_cols_list = []
    build_cols_list = []
    interval_ops_list = []
    for i, var_name in enumerate(rtjf_states_vars_names):
        # Generate list information for interval filters.
        build_indices = rtjf_build_indices_list[i]
        probe_cols = np.array([x[0] for x in rtjf_interval_cols[i]], dtype=np.int64)
        probe_cols_list.append(probe_cols)
        build_cols = np.array(
            [build_indices[x[1]] for x in rtjf_interval_cols[i]], dtype=np.int64
        )
        build_cols_list.append(build_cols)
        ops = pd.array([x[2] for x in rtjf_interval_cols[i]], dtype="string")
        interval_ops_list.append(ops)
        if len(rtjf_cols[i]) > 0:
            # Fetch the precision and time zone for each of the used columns
            precisions, time_zones = get_rtjf_cols_extra_info(col_typs, rtjf_cols[i])
            # Get runtime join filter column min/max map
            rtjf_str += f"  filtered_cols, bounds = get_rtjf_col_min_max_unique_map({var_name}, np.arange(len(rtjf_cols_{call_id}[{i}])), rtjf_cols_{call_id}[{i}], used_cols_{call_id}, {precisions}, {time_zones})\n"
            # Add runtime join filters to Iceberg file scan filters
            rtjf_str += f"  iceberg_filters = add_rtjf_iceberg_filter({var_name}, iceberg_filters, filtered_cols, equality_ops_{call_id}[{i}], bounds)\n"
            # Add runtime join filters to Iceberg expression filters for Arrow data filtering
            rtjf_str += f"  rtjf_expr_{i} = gen_runtime_join_filter_expr(filtered_cols, equality_ops_{call_id}[{i}], bounds, {time_zones})\n"
            rtjf_str += f"  if rtjf_expr_{i} != '':\n"
            rtjf_str += f"    rtjf_exprs.append(rtjf_expr_{i})\n"
        if len(rtjf_interval_cols[i]) > 0:
            # Fetch the precision and time zone for each of the used columns
            precisions, time_zones = get_rtjf_cols_extra_info(col_typs, probe_cols)
            # Get runtime join filter column min/max map
            rtjf_str += f"  filtered_cols, bounds = get_rtjf_col_min_max_unique_map({var_name}, build_cols_{call_id}[{i}], probe_cols_{call_id}[{i}], used_cols_{call_id}, {precisions}, {time_zones})\n"
            # Add runtime join filters to Iceberg file scan filters
            rtjf_str += f"  iceberg_filters = add_rtjf_iceberg_filter({var_name}, iceberg_filters, filtered_cols, interval_ops_{call_id}[{i}], bounds)\n"
            # Add runtime join filters to Iceberg expression filters for Arrow data filtering
            rtjf_str += f"  rtjf_expr_{i} = gen_runtime_join_filter_expr(filtered_cols, interval_ops_{call_id}[{i}], bounds, {time_zones})\n"
            rtjf_str += f"  if rtjf_expr_{i} != '':\n"
            rtjf_str += f"    rtjf_exprs.append(rtjf_expr_{i})\n"
    rtjf_str += f"  combined_iceberg_expr_filter_f_str = (iceberg_expr_filter_f_str_{call_id})\n"
    if len(rtjf_states_vars_names):
        rtjf_str += '  rtjf_expr = f"({\' & \'.join(rtjf_exprs)})" if len(rtjf_exprs) else "True"\n'
        rtjf_str += f'  combined_iceberg_expr_filter_f_str +=  " & " if len(iceberg_expr_filter_f_str_{call_id}) else ""\n'
        rtjf_str += "  combined_iceberg_expr_filter_f_str += rtjf_expr\n"
        if bodo.user_logging.get_verbose_level() >= 2:
            rtjf_str += "  log_message('Iceberg I/O', f'Runtime join filter expression: {rtjf_expr}')\n"

    func_text = (
        f"def sql_reader_chunked_py(table_id, conn, {filter_args}{comma} {','.join(rtjf_states_vars_names)}):\n"
        f"  ev = bodo.utils.tracing.Event('read_iceberg', {parallel})\n"
        f'  iceberg_filters = get_filters_pyobject("{filter_str}", ({filter_args}{comma}))\n'
        f"  filter_scalars_pyobject = get_filter_scalars_pyobject(({filter_args}{comma}))\n"
        f"{rtjf_str}"
        # Iceberg C++ Parquet Reader
        f"  iceberg_reader = iceberg_pq_reader_init_py_entry(\n"
        f"    unicode_to_utf8(conn),\n"
        f"    unicode_to_utf8(table_id),\n"
        f"    {parallel},\n"
        f"    {-1 if limit is None else limit},\n"
        f"    iceberg_filters,\n"
        f"    unicode_to_utf8(combined_iceberg_expr_filter_f_str),\n"
        f"    filter_scalars_pyobject,\n"
        f"    selected_cols_arr_{call_id}.ctypes,\n"
        f"    {len(source_selected_cols)},\n"
        f"    nullable_cols_arr_{call_id}.ctypes,\n"
        f"    source_pyarrow_schema_{call_id},\n"
        f"    {dict_str_cols_str},\n"
        f"    {dict_encode_in_bodo},\n"  # create_dict_from_string
        f"    {chunksize},\n"
        f"    {sql_op_id},\n"
        f"    out_type,\n"
        f"  )\n"
        f"  return iceberg_reader\n"
    )

    glbls = globals().copy()  # TODO: fix globals after Numba's #3355 is resolved
    glbls.update(
        {
            "objmode": bodo.no_warning_objmode,
            "unicode_to_utf8": unicode_to_utf8,
            "iceberg_pq_reader_init_py_entry": iceberg_pq_reader_init_py_entry,
            "get_filters_pyobject": get_filters_pyobject,
            "get_filter_scalars_pyobject": bodo.io.parquet_pio.get_filter_scalars_pyobject,
            f"iceberg_expr_filter_f_str_{call_id}": iceberg_expr_filter_f_str,
            "out_type": ArrowReaderType(col_names, col_typs),
            f"selected_cols_arr_{call_id}": np.array(source_selected_cols, np.int32),
            f"nullable_cols_arr_{call_id}": np.array(nullable_cols, np.int32),
            f"dict_str_cols_arr_{call_id}": np.array(str_as_dict_cols, np.int32),
            f"source_pyarrow_schema_{call_id}": source_pyarrow_schema,
            f"used_cols_{call_id}": used_cols,
            f"rtjf_cols_{call_id}": rtjf_cols,
            f"equality_ops_{call_id}": [
                pd.array(["=="] * len(rtjf_cols[i]), dtype="string")
                for i in range(len(rtjf_cols))
            ],
            f"interval_ops_{call_id}": interval_ops_list,
            f"build_cols_{call_id}": build_cols_list,
            f"probe_cols_{call_id}": probe_cols_list,
            "log_message": bodo.user_logging.log_message,
        }
    )
    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    sql_reader_py = loc_vars["sql_reader_chunked_py"]

    # TODO: no_cpython_wrapper=True crashes for some reason
    jit_func = numba.njit(sql_reader_py)
    compiled_funcs.append(jit_func)
    return jit_func


def _gen_iceberg_reader_py(
    col_names: list[str],
    col_typs: list[pt.Any],
    index_column_name: str | None,
    index_column_type,
    out_used_cols: list[int],
    limit: int | None,
    parallel: bool,
    typemap,
    filters: pt.Any | None,
    pyarrow_schema: pa.Schema | None,
    is_dead_table: bool,
    is_merge_into: bool,
    dict_encode_in_bodo: bool,
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
        typingctx: Typing context used for compiling code.
        targetctx: Target context used for compiling code.
        db_type: Type of SQL source used to distinguish between backends.
        limit: Does the query contain a limit? This is only used to divide
            data with regular SQL.
        parallel: Is the implementation parallel?
        typemap: Maps variables name -> types. Used by iceberg for filters.
        filters: DNF Filter info used by iceberg to generate runtime filters.
            This should only be used for Iceberg.
        pyarrow_schema: PyArrow schema for the source table
        is_merge_into: Does this query result from a merge into query? If so
            this limits the filtering we can do with Iceberg as we
            must load entire files.
        dict_encode_in_bodo: Whether the dict-encoding should be done in Bodo
            instead of Arrow.
    """
    # a unique int used to create global variables with unique names
    call_id = next_label()

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
    py_table_type = types.none if is_dead_table else TableType(tuple(col_typs))

    # Handle filter information because we may need to update the function header
    filter_args = ""
    filter_map = {}
    if filters:
        filter_map, _ = bodo.ir.connector.generate_filter_map(filters)
        filter_args = ", ".join(filter_map.values())

    assert pyarrow_schema is not None, (
        "SQLNode must contain a pyarrow_schema if reading from an Iceberg database"
    )

    # Generate the predicate filters. Note we pass
    # all col names as possible partitions via partition names.
    # The expression filters are returned as f-strings so that we can
    # pass them to the runtime to generate the filters dynamically
    # for the various schemas (to account for schema evolution).
    iceberg_expr_filter_f_str = bodo.ir.connector.generate_arrow_filters(
        filters,
        filter_map,
        col_names,
        col_names,
        col_typs,
        typemap,
        "iceberg",
        output_expr_filters_as_f_string=True,
    )
    filter_str = filters_to_iceberg_expr(filters, filter_map)

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
        int(is_nullable_ignore_sentinels(col_typs[i])) for i in selected_cols
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

    # Generate a temporary one for codegen:
    comma = "," if filter_args else ""
    func_text = (
        f"def sql_reader_py(table_id, conn, {filter_args}):\n"
        f"  ev = bodo.utils.tracing.Event('read_iceberg', {parallel})\n"
        f'  iceberg_filters = get_filters_pyobject("{filter_str}", ({filter_args}{comma}))\n'
        f"  filter_scalars_pyobject = get_filter_scalars_pyobject(({filter_args}{comma}))\n"
        # Iceberg C++ Parquet Reader
        f"  out_table, total_rows, file_list, snapshot_id = iceberg_pq_read_py_entry(\n"
        f"    unicode_to_utf8(conn),\n"
        f"    unicode_to_utf8(table_id),\n"
        f"    {parallel},\n"
        f"    {-1 if limit is None else limit},\n"
        f"    iceberg_filters,\n"
        f"    unicode_to_utf8(iceberg_expr_filter_f_str_{call_id}),\n"
        f"    filter_scalars_pyobject,\n"
        #     TODO Confirm that we're computing selected_cols correctly
        f"    selected_cols_arr_{call_id}.ctypes,\n"
        f"    {len(selected_cols)},\n"
        #     TODO Confirm that we're computing is_nullable correctly
        f"    nullable_cols_arr_{call_id}.ctypes,\n"
        f"    pyarrow_schema_{call_id},\n"
        f"    {dict_str_cols_str},\n"
        f"    {dict_encode_in_bodo},\n"  # create_dict_from_string
        f"    {is_merge_into},\n"
        f"  )\n"
    )

    # Mostly copied over from _gen_pq_reader_py
    # TODO XXX Refactor?

    # Compute number of rows stored on rank for head optimization. See _gen_pq_reader_py
    if parallel:
        func_text += "  local_rows = get_node_portion(total_rows, bodo.get_size(), bodo.get_rank())\n"
    else:
        func_text += "  local_rows = total_rows\n"

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
            func_text += "  table_var = set_table_len(table_var, local_rows)\n"

    # Handle index column
    index_var = "None"

    # Since we don't support `index_col`` with iceberg yet, we can't test this yet.
    if index_column_name is not None:  # pragma: no cover
        # The index column is defined by the SQLReader to always be placed at the end of the query.
        index_arr_ind = (len(out_used_cols) + 1) if not is_dead_table else 0
        index_var = f"array_from_cpp_table(out_table, {index_arr_ind}, index_col_typ)"

    func_text += f"  index_var = {index_var}\n"

    func_text += "  delete_table(out_table)\n"
    func_text += "  ev.finalize()\n"
    func_text += "  return (total_rows, table_var, index_var, file_list, snapshot_id)\n"
    glbls = globals()  # TODO: fix globals after Numba's #3355 is resolved
    glbls.update(
        {
            "bodo": bodo,
            "objmode": bodo.no_warning_objmode,
            f"py_table_type_{call_id}": py_table_type,
            "index_col_typ": index_column_type,
            f"table_idx_{call_id}": table_idx,
            f"pyarrow_schema_{call_id}": pyarrow_schema,
            "unicode_to_utf8": unicode_to_utf8,
            "check_and_propagate_cpp_exception": check_and_propagate_cpp_exception,
            "array_from_cpp_table": array_from_cpp_table,
            "delete_table": delete_table,
            "cpp_table_to_py_table": cpp_table_to_py_table,
            "set_table_len": bodo.hiframes.table.set_table_len,
            "get_node_portion": bodo.libs.distributed_api.get_node_portion,
            f"selected_cols_arr_{call_id}": np.array(selected_cols, np.int32),  # type: ignore
            f"nullable_cols_arr_{call_id}": np.array(nullable_cols, np.int32),  # type: ignore
            f"dict_str_cols_arr_{call_id}": np.array(str_as_dict_cols, np.int32),  # type: ignore
            "get_filters_pyobject": get_filters_pyobject,
            f"iceberg_expr_filter_f_str_{call_id}": iceberg_expr_filter_f_str,
            "get_filter_scalars_pyobject": bodo.io.parquet_pio.get_filter_scalars_pyobject,
            "iceberg_pq_read_py_entry": iceberg_pq_read_py_entry,
        }
    )

    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    sql_reader_py = loc_vars["sql_reader_py"]

    # TODO: no_cpython_wrapper=True crashes for some reason
    jit_func = numba.njit(sql_reader_py)
    compiled_funcs.append(jit_func)
    return jit_func


numba.parfors.array_analysis.array_analysis_extensions[IcebergReader] = (
    bodo.ir.connector.connector_array_analysis
)
distributed_analysis.distributed_analysis_extensions[IcebergReader] = (
    bodo.ir.connector.connector_distributed_analysis
)
typeinfer.typeinfer_extensions[IcebergReader] = bodo.ir.connector.connector_typeinfer
ir_utils.visit_vars_extensions[IcebergReader] = bodo.ir.connector.visit_vars_connector
ir_utils.remove_dead_extensions[IcebergReader] = remove_dead_iceberg
numba.core.analysis.ir_extension_usedefs[IcebergReader] = (
    bodo.ir.connector.connector_usedefs
)
ir_utils.copy_propagate_extensions[IcebergReader] = (
    bodo.ir.connector.get_copies_connector
)
ir_utils.apply_copy_propagate_extensions[IcebergReader] = (
    bodo.ir.connector.apply_copies_connector
)
ir_utils.build_defs_extensions[IcebergReader] = (
    bodo.ir.connector.build_connector_definitions
)
distributed_pass.distributed_run_extensions[IcebergReader] = iceberg_distributed_run
remove_dead_column_extensions[IcebergReader] = iceberg_remove_dead_column
ir_extension_table_column_use[IcebergReader] = (
    bodo.ir.connector.connector_table_column_use
)
