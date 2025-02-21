"""IR node for the parquet data access"""

from typing import TYPE_CHECKING

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from llvmlite import ir as lir
from numba.core import cgutils, ir, ir_utils, typeinfer, types
from numba.core.ir_utils import (
    compile_to_numba_ir,
    get_definition,
    guard,
    mk_unique_var,
    next_label,
    replace_arg_nodes,
)
from numba.extending import (
    NativeValue,
    intrinsic,
    models,
    register_model,
    unbox,
)

import bodo
import bodo.ir.connector
import bodo.user_logging
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.table import Table, TableType  # noqa
from bodo.io import arrow_cpp  # type: ignore
from bodo.io.arrow_reader import ArrowReaderType
from bodo.io.fs_io import (
    get_storage_options_pyobject,
    storage_options_dict_type,
)
from bodo.io.helpers import numba_to_pyarrow_schema, pyarrow_schema_type
from bodo.io.parquet_pio import (
    ParquetFileInfo,
    get_filters_pyobject,
    parquet_file_schema,
    parquet_predicate_type,
)
from bodo.ir.connector import Connector
from bodo.libs.array import (
    array_from_cpp_table,
    cpp_table_to_py_table,
    delete_table,
    table_type,
)
from bodo.libs.dict_arr_ext import dict_str_arr_type
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.transforms.table_column_del_pass import (
    ir_extension_table_column_use,
    remove_dead_column_extensions,
)
from bodo.utils.transform import get_const_value
from bodo.utils.typing import (
    BodoError,
    FilenameType,
    is_nullable_ignore_sentinels,
)
from bodo.utils.utils import (
    bodo_exec,
    check_and_propagate_cpp_exception,
    create_arg_hash,
    inlined_check_and_propagate_cpp_exception,
    numba_to_c_type,
    sanitize_varname,
)

if TYPE_CHECKING:  # pragma: no cover
    from llvmlite.ir.builder import IRBuilder
    from numba.core.base import BaseContext


ll.add_symbol("pq_read_py_entry", arrow_cpp.pq_read_py_entry)
ll.add_symbol("pq_reader_init_py_entry", arrow_cpp.pq_reader_init_py_entry)


class ReadParquetFilepathType(types.Opaque):
    """Type for file path object passed to C++. It is just a Python object passed
    as a pointer to C++ (can be Python list of strings or Python string)
    """

    def __init__(self):
        super().__init__(name="ReadParquetFilepathType")


read_parquet_fpath_type = ReadParquetFilepathType()
types.read_parquet_fpath_type = read_parquet_fpath_type  # type: ignore
register_model(ReadParquetFilepathType)(models.OpaqueModel)


@unbox(ReadParquetFilepathType)
def unbox_read_parquet_fpath_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return NativeValue(val)


class ParquetHandler:
    """analyze and transform parquet IO calls"""

    def __init__(self, func_ir: ir.FunctionIR, typingctx, args, _locals):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.args = args
        self.locals = _locals

    def gen_parquet_read(
        self,
        file_name,
        lhs: ir.Var,
        columns,
        storage_options=None,
        input_file_name_col=None,
        read_as_dict_cols=None,
        use_hive=True,
        _bodo_read_as_table=False,
        chunksize: int | None = None,
        use_index: bool = True,
        sql_op_id: int = -1,
    ):
        scope = lhs.scope
        loc = lhs.loc
        table_types = None
        if lhs.name in self.locals:
            table_types = self.locals[lhs.name]
            self.locals.pop(lhs.name)

        convert_types = {}
        # user-specified type conversion
        if (lhs.name + ":convert") in self.locals:
            convert_types = self.locals[lhs.name + ":convert"]
            self.locals.pop(lhs.name + ":convert")

        if table_types is None:
            msg = (
                "Parquet schema not available. Either path argument "
                "should be constant for Bodo to look at the file at compile "
                "time or schema should be provided. For more information, "
                "see: https://docs.bodo.ai/latest/file_io/#parquet-section."
            )
            file_name_str = get_const_value(
                file_name,
                self.func_ir,
                msg,
                arg_types=self.args,
                file_info=ParquetFileInfo(
                    columns,
                    storage_options=storage_options,
                    input_file_name_col=input_file_name_col,
                    read_as_dict_cols=read_as_dict_cols,
                    use_hive=use_hive,
                ),
            )

            # get_const_value forces variable to be literal which should convert it to
            # FilenameType. If so, the schema will be part of the type
            var_def = guard(get_definition, self.func_ir, file_name)
            if isinstance(var_def, ir.Arg) and isinstance(
                self.args[var_def.index], FilenameType
            ):
                typ: FilenameType = self.args[var_def.index]
                (
                    col_names,
                    col_types,
                    index_col,
                    col_indices,
                    partition_names,
                    unsupported_columns,
                    unsupported_arrow_types,
                    arrow_schema,
                ) = typ.schema
            else:
                (
                    col_names,
                    col_types,
                    index_col,
                    col_indices,
                    partition_names,
                    unsupported_columns,
                    unsupported_arrow_types,
                    arrow_schema,
                ) = parquet_file_schema(
                    file_name_str,
                    columns,
                    storage_options,
                    input_file_name_col,
                    read_as_dict_cols,
                    use_hive,
                )
        else:
            all_col_names: list[str] = list(table_types.keys())
            # Create a map for efficient index lookup
            all_col_names_map = {c: i for i, c in enumerate(all_col_names)}
            col_types_total = list(table_types.values())

            # TODO: allow specifying types of only selected columns
            col_names: list[str] = all_col_names if columns is None else columns
            col_indices = [all_col_names_map[c] for c in col_names]
            col_types = [col_types_total[all_col_names_map[c]] for c in col_names]

            # We currently assume, when locals is provided, that the index
            # column is a Parquet column of the form `__index__level__{i}__`
            # for some integer i. This was Arrow's standard behavior for a
            # while; newer versions added options to specify the index in
            # metadata only. Need to investigate and/or have another parameter
            # TODO: https://bodo.atlassian.net/browse/BE-4110
            index_col = next(
                (x for x in col_names if x.startswith("__index_level_")), None
            )

            partition_names = []
            # If a user provides the schema, all types must be valid Bodo types.
            unsupported_columns = []
            unsupported_arrow_types = []
            arrow_schema = numba_to_pyarrow_schema(
                DataFrameType(data=tuple(col_types), columns=tuple(col_names)),
            )

        index_colname = (
            None if (isinstance(index_col, dict) or index_col is None) else index_col
        )
        # If we have an index column, remove it from the type to simplify the table.
        index_column_index = None
        index_column_type = types.none
        if index_colname:
            type_index = col_names.index(index_colname)
            index_column_index = col_indices.pop(type_index)
            index_column_type = col_types.pop(type_index)
            col_names.pop(type_index)

        # HACK convert types using decorator for int columns with NaN
        for i, c in enumerate(col_names):
            if c in convert_types:
                col_types[i] = convert_types[c]

        if chunksize is None:
            data_arrs = [
                ir.Var(scope, mk_unique_var("pq_table"), loc),
                ir.Var(scope, mk_unique_var("pq_index"), loc),
            ]
        else:
            data_arrs = [ir.Var(lhs.scope, mk_unique_var("arrow_iterator"), lhs.loc)]

        nodes = [
            ParquetReader(
                file_name,
                lhs.name,
                col_names,
                col_indices,
                col_types,
                data_arrs,
                loc,
                partition_names,
                storage_options,
                index_column_index if use_index else None,
                index_column_type if use_index else types.none,
                input_file_name_col,
                unsupported_columns,
                unsupported_arrow_types,
                arrow_schema,
                use_hive,
                chunksize,
                sql_op_id,
            )
        ]

        return col_names, data_arrs, index_col, nodes, col_types, index_column_type


class ParquetReader(Connector):
    connector_typ = "parquet"

    def __init__(
        self,
        file_name,
        df_out_varname: str,
        col_names: list[str],
        col_indices,
        out_table_col_types: list[types.ArrayCompatible],
        out_vars: list[ir.Var],
        loc: ir.Loc,
        partition_names,
        # These are the same storage_options that would be passed to pandas
        storage_options,
        index_column_index,
        index_column_type,
        input_file_name_col,
        unsupported_columns,
        unsupported_arrow_types,
        arrow_schema: pa.Schema,
        use_hive: bool,
        # Batch size to read chunks in, or none, to read the entire table together
        # Treated as compile-time constant for simplicity
        # But not enforced that all chunks are this size
        chunksize: int | None = None,
        # Operator ID generated by BodoSQL for query profile
        # purposes. Only supported in the streaming case.
        sql_op_id: int = -1,
    ):
        # From Base Connector Class
        self.out_table_col_names = col_names
        self.out_table_col_types = out_table_col_types

        self.file_name = file_name
        self.df_out_varname = df_out_varname  # used only for printing
        self.col_indices = col_indices
        # Original out types + columns are maintained even if columns are pruned.
        # This is maintained in case we need type info for filter pushdown and
        # the column has been eliminated.
        # For example, if our Pandas code was:
        # def ex(filename):
        #     df = pd.read_parquet(filename)
        #     df = df[df.A > 1]
        #     return df[["B", "C"]]
        # Then DCE should remove all columns from out_table_col_names/out_types except B and C,
        # but we still need to the type of column A to determine if we need to generate
        # a cast inside the arrow filters.
        self.original_table_col_types = out_table_col_types
        self.original_df_colnames = col_names
        self.out_vars = out_vars
        self.loc = loc
        self.partition_names = partition_names
        self.filters = None
        # storage_options passed to pandas during read_parquet
        self.storage_options = storage_options
        self.index_column_index = index_column_index
        self.index_column_type = index_column_type
        # Columns within the output table type that are actually used.
        # These will be updated during optimzations and do not contain
        # the actual columns numbers that should be loaded. For more
        # information see 'pq_remove_dead_column'.
        self.out_used_cols = list(range(len(col_indices)))
        # Name of the column where we insert the name of file that the row comes from
        self.input_file_name_col = input_file_name_col
        # These fields are used to enable compilation if unsupported columns
        # get eliminated.
        self.unsupported_columns = unsupported_columns
        self.unsupported_arrow_types = unsupported_arrow_types
        self.arrow_schema = arrow_schema
        # Is the variable currently alive. This should be replaced with more
        # robust handling in connectors.
        self.is_live_table = True
        self.use_hive = use_hive
        self.chunksize = chunksize
        self.sql_op_id = sql_op_id

    def __repr__(self):  # pragma: no cover
        # TODO
        return f"({self.df_out_varname}) = ReadParquet({self.file_name.name}, {self.out_table_col_names}, {self.col_indices}, {self.out_table_col_types}, {self.original_table_col_types}, {self.original_df_colnames}, {self.out_vars}, {self.partition_names}, {self.filters}, {self.storage_options}, {self.index_column_index}, {self.index_column_type}, {self.out_used_cols}, {self.input_file_name_col}, {self.unsupported_columns}, {self.unsupported_arrow_types}, {self.arrow_schema}, chunksize={self.chunksize}, sql_op_id={self.sql_op_id})"

    def out_vars_and_types(self) -> list[tuple[str, types.Type]]:
        return (
            [
                (
                    self.out_vars[0].name,
                    ArrowReaderType(self.out_table_col_names, self.out_table_col_types),
                )
            ]
            if self.is_streaming
            else [
                (self.out_vars[0].name, TableType(tuple(self.out_table_col_types))),
                (self.out_vars[1].name, self.index_column_type),
            ]
        )


def remove_dead_pq(
    pq_node: ParquetReader,
    lives_no_aliases,
    lives,
    arg_aliases,
    alias_map,
    func_ir,
    typemap,
):
    """
    Function that eliminates parquet reader variables when they
    are no longer live.
    """
    if pq_node.is_streaming:
        return pq_node

    table_var = pq_node.out_vars[0].name
    index_var = pq_node.out_vars[1].name
    if table_var not in lives and index_var not in lives:
        # If neither the table or index is live, remove the node.
        return None
    elif table_var not in lives:
        # If table isn't live we only want to load the index.
        # To do this we should mark the col_indices as empty
        pq_node.col_indices = []
        pq_node.out_table_col_names = []
        pq_node.out_used_cols = []
        pq_node.is_live_table = False

    elif index_var not in lives:
        # If the index_var not in lives we don't load the index.
        # To do this we mark the index_column_index as None
        pq_node.index_column_index = None
        pq_node.index_column_type = types.none

    # TODO: Update the usecols if only 1 of the variables is live.
    return pq_node


def pq_remove_dead_column(pq_node: ParquetReader, column_live_map, equiv_vars, typemap):
    """
    Function that tracks which columns to prune from the Parquet node.
    This updates out_used_cols which stores which arrays in the
    types will need to actually be loaded.

    This is mapped to the actual file columns in during distributed pass.
    """
    return bodo.ir.connector.base_connector_remove_dead_columns(
        pq_node,
        column_live_map,
        equiv_vars,
        typemap,
        "ParquetReader",
        # col_indices is set to an empty list if the table is dead
        # see 'remove_dead_pq'
        pq_node.col_indices,
        # Parquet can track length without loading any columns.
        require_one_column=False,
    )


def pq_distributed_run(
    pq_node: ParquetReader,
    array_dists,
    typemap,
    calltypes,
    typingctx,
    targetctx,
    is_independent: bool = False,  # is_independent currently only used for sql_distributed_run for Snowflake
    meta_head_only_info=None,
):
    """lower ParquetReader into regular Numba nodes. Generates code for Parquet
    data read.
    """
    n_cols = len(pq_node.out_vars)
    filter_str = "None"

    filter_map, filter_vars = bodo.ir.connector.generate_filter_map(pq_node.filters)
    extra_args = ", ".join(filter_map.values())
    filter_str = bodo.ir.connector.generate_arrow_filters(
        pq_node.filters,
        filter_map,
        pq_node.original_df_colnames,
        pq_node.partition_names,
        pq_node.original_table_col_types,
        typemap,
        "parquet",
        output_expr_filters_as_f_string=False,
        sql_op_id=pq_node.sql_op_id,
    )
    arg_names = ", ".join(f"out{i}" for i in range(n_cols))
    func_text = f"def pq_impl(fname, {extra_args}):\n"
    # total_rows is used for setting total size variable below
    if pq_node.chunksize is None:
        func_text += (
            f"    (total_rows, {arg_names},) = _pq_reader_py(fname, {extra_args})\n"
        )
    else:
        func_text += f"    pq_reader = _pq_reader_py(fname, {extra_args})\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    pq_impl = loc_vars["pq_impl"]

    # Add debug info about column pruning and dictionary encoded arrays.
    if bodo.user_logging.get_verbose_level() >= 1:
        out_types = pq_node.out_table_col_types
        # State which columns are pruned
        pq_source = pq_node.loc.strformat()
        pq_cols = []
        dict_encoded_cols = []
        for i in pq_node.out_used_cols:
            colname = pq_node.out_table_col_names[i]
            pq_cols.append(colname)
            if isinstance(out_types[i], bodo.libs.dict_arr_ext.DictionaryArrayType):
                dict_encoded_cols.append(colname)
        op_id_msg = (
            f" (Operator ID: {pq_node.sql_op_id}) " if pq_node.sql_op_id != -1 else ""
        )
        pruning_msg = (
            "Finish column pruning on read_parquet node%s:\n%s\nColumns loaded %s\n"
        )
        bodo.user_logging.log_message(
            "Column Pruning",
            pruning_msg,
            op_id_msg,
            pq_source,
            pq_cols,
        )
        # Log if any columns use dictionary encoded arrays.
        if dict_encoded_cols:
            encoding_msg = "Finished optimized encoding on read_parquet node%s:\n%s\nColumns %s using dictionary encoding to reduce memory usage.\n"
            bodo.user_logging.log_message(
                "Dictionary Encoding",
                encoding_msg,
                op_id_msg,
                pq_source,
                dict_encoded_cols,
            )

    # Parallel read flag
    if pq_node.is_streaming:
        parallel = bodo.ir.connector.is_chunked_connector_table_parallel(
            pq_node, array_dists, "ParquetReader"
        )
    else:
        parallel = bodo.ir.connector.is_connector_table_parallel(
            pq_node, array_dists, typemap, "ParquetReader"
        )

    # Check for any unsupported columns still remaining
    if pq_node.unsupported_columns:
        used_cols_set = set(pq_node.out_used_cols)
        unsupported_cols_set = set(pq_node.unsupported_columns)
        remaining_unsupported = used_cols_set & unsupported_cols_set
        if remaining_unsupported:
            unsupported_list = sorted(remaining_unsupported)
            msg_list = [
                "pandas.read_parquet(): 1 or more columns found with Arrow types that are not supported in Bodo and could not be eliminated. "
                + "Please manually remove these columns from your read_parquet with the 'columns' argument. If these "
                + "columns are needed, you will need to modify your dataset to use a supported type.",
                "Unsupported Columns:",
            ]
            # Find the arrow types for the unsupported types
            idx = 0
            for col_num in unsupported_list:
                while pq_node.unsupported_columns[idx] != col_num:
                    idx += 1
                msg_list.append(
                    f"Column '{pq_node.out_table_col_names[col_num]}' with unsupported arrow type {pq_node.unsupported_arrow_types[idx]}"
                )
                idx += 1
            total_msg = "\n".join(msg_list)
            raise BodoError(total_msg, loc=pq_node.loc)

    genargs = (
        pq_node.out_table_col_names,
        pq_node.col_indices,
        pq_node.out_used_cols,
        pq_node.out_table_col_types,
        pq_node.storage_options,
        pq_node.partition_names,
        filter_str,
        extra_args,
        parallel,
        meta_head_only_info,
        pq_node.index_column_index,
        pq_node.index_column_type,
        pq_node.input_file_name_col,
        not pq_node.is_live_table,
        pq_node.arrow_schema,
        pq_node.use_hive,
    )

    if pq_node.chunksize is None:
        pq_reader_py = _gen_pq_reader_py(*genargs)
    else:
        pq_reader_py = _gen_pq_reader_chunked_py(
            *genargs, pq_node.chunksize, pq_node.sql_op_id
        )

    # First arg is the path to the parquet dataset, and can be a string or a list
    # of strings
    fname_type = typemap[pq_node.file_name.name]
    arg_types = (fname_type,) + tuple(typemap[v.name] for v in filter_vars)
    f_block = compile_to_numba_ir(
        pq_impl,
        {"_pq_reader_py": pq_reader_py},
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=arg_types,
        typemap=typemap,
        calltypes=calltypes,
    ).blocks.popitem()[1]
    replace_arg_nodes(f_block, [pq_node.file_name] + filter_vars)
    nodes = f_block.body[:-3]
    # set total size variable if necessary (for limit pushdown)
    # value comes from 'total_rows' output of '_pq_reader_py' above
    if meta_head_only_info:
        nodes[-3].target = meta_head_only_info[1]

    # In the streaming case, ParquetReader only return an ArrowReader object
    # Thus, we only need to pair 1 out_var to the last target node
    # We don't need to pair any other elements
    if pq_node.is_streaming:
        nodes[-1].target = pq_node.out_vars[0]
        return nodes

    # assign output table
    nodes[-2].target = pq_node.out_vars[0]
    # assign output index array
    nodes[-1].target = pq_node.out_vars[1]
    # At most one of the table and the index
    # can be dead because otherwise the whole
    # node should have already been removed.
    assert not (pq_node.index_column_index is None and not pq_node.is_live_table), (
        "At most one of table and index should be dead if the Parquet IR node is live"
    )
    if pq_node.index_column_index is None:
        # If the index_col is dead, remove the node.
        nodes.pop(-1)
    elif not pq_node.is_live_table:
        # If the table is dead, remove the node
        nodes.pop(-2)

    return nodes


def pq_reader_params(
    meta_head_only_info: tuple | None,
    col_names: list[str],
    col_indices,
    partition_names,
    input_file_name_col,
    out_used_cols,
    index_column_index,
    index_column_type,
    out_types: list[types.ArrayCompatible],
):
    # head-only optimization: we may need to read only the first few rows
    tot_rows_to_read = -1  # read all rows by default
    if meta_head_only_info and meta_head_only_info[0] is not None:
        tot_rows_to_read = meta_head_only_info[0]

    # NOTE: col_indices are the indices of columns in the parquet file (not in
    # the output of read_parquet)

    sanitized_col_names = [sanitize_varname(c) for c in col_names]
    partition_names = [sanitize_varname(c) for c in partition_names]

    # If the input_file_name column was pruned out, then set it to None
    # (since that's what it effectively is now). Otherwise keep it
    # (and sanitize the variable name)
    # NOTE We could modify the ParquetReader node to store the
    # index instead of the name of the column to have slightly
    # cleaner code, although we need to make sure dead column elimination
    # works as expected.
    input_file_name_col_out = (
        sanitize_varname(input_file_name_col)
        if (input_file_name_col is not None)
        and (col_names.index(input_file_name_col) in out_used_cols)
        else None
    )

    # Create maps for efficient index lookups.
    col_indices_map = {c: i for i, c in enumerate(col_indices)}
    sanitized_col_names_map = {c: i for i, c in enumerate(sanitized_col_names)}

    # Get list of selected columns to pass to C++ (not including partition
    # columns, since they are not in the parquet files).
    # C++ doesn't need to know the order of output columns, and to simplify
    # the code we will pass the indices of columns in the parquet file sorted.
    # C++ code will add partition columns to the end of its output table.
    # Here because columns may have been eliminated by 'pq_remove_dead_column',
    # we only load the indices in out_used_cols.
    selected_cols = []
    partition_indices = set()
    cols_to_skip = partition_names + [input_file_name_col]
    for i in out_used_cols:
        if sanitized_col_names[i] not in cols_to_skip:
            selected_cols.append(col_indices[i])
        elif (not input_file_name_col) or (
            sanitized_col_names[i] != input_file_name_col
        ):
            # Track which partitions are valid to simplify filtering later
            partition_indices.add(col_indices[i])

    if index_column_index is not None:
        selected_cols.append(index_column_index)
    selected_cols = sorted(selected_cols)
    selected_cols_map = {c: i for i, c in enumerate(selected_cols)}

    # Tell C++ which columns in the parquet file are nullable, since there
    # are some types like integer which Arrow always considers to be nullable
    # but pandas might not. This is mainly intended to tell C++ which Int/Bool
    # arrays require null bitmap and which don't.
    # We need to load the nullable check in the same order as select columns. To do
    # this, we first need to determine the index of each selected column in the original
    # type and check if that type is nullable.
    nullable_cols = [
        (
            int(is_nullable_ignore_sentinels(out_types[col_indices_map[col_in_idx]]))
            if col_in_idx != index_column_index
            else int(is_nullable_ignore_sentinels(index_column_type))
        )
        for col_in_idx in selected_cols
    ]

    # pass indices to C++ of the selected string columns that are to be read
    # in dictionary-encoded format
    str_as_dict_cols = []
    for col_in_idx in selected_cols:
        if col_in_idx == index_column_index:
            t = index_column_type
        else:
            t = out_types[col_indices_map[col_in_idx]]
        if t == dict_str_arr_type:
            str_as_dict_cols.append(col_in_idx)

    # partition_names is the list of *all* partition column names in the
    # parquet dataset as given by pyarrow.parquet.ParquetDataset.
    # We pass selected partition columns to C++, in the order and index used
    # by pyarrow.parquet.ParquetDataset (e.g. 0 is the first partition col)
    # We also pass the dtype of categorical codes
    sel_partition_names = []
    # Create a map for efficient index lookup
    sel_partition_names_map = {}
    selected_partition_cols = []
    partition_col_cat_dtypes = []
    for i, part_name in enumerate(partition_names):
        try:
            col_out_idx = sanitized_col_names_map[part_name]
            # Only load part_name values that are selected
            # This occurs if we can prune these columns.
            if col_indices[col_out_idx] not in partition_indices:
                # this partition column has not been selected for read
                continue
        except (KeyError, ValueError):
            # this partition column has not been selected for read
            # This occurs when the user provides columns
            continue
        sel_partition_names_map[part_name] = len(sel_partition_names)
        sel_partition_names.append(part_name)
        selected_partition_cols.append(i)
        part_col_type = out_types[col_out_idx].dtype
        cat_int_dtype = bodo.hiframes.pd_categorical_ext.get_categories_int_type(
            part_col_type
        )
        partition_col_cat_dtypes.append(numba_to_c_type(cat_int_dtype))

    return (
        tot_rows_to_read,
        col_indices_map,
        sanitized_col_names_map,
        input_file_name_col_out,
        selected_cols,
        selected_cols_map,
        selected_partition_cols,
        partition_col_cat_dtypes,
        str_as_dict_cols,
        nullable_cols,
        sel_partition_names,
        partition_indices,
        sanitized_col_names,
        sel_partition_names_map,
    )


def _gen_pq_reader_py(
    col_names: list[str],
    col_indices,
    out_used_cols,
    out_types: list[types.ArrayCompatible],
    storage_options,
    partition_names,
    expr_filter_str: str,
    extra_args,
    is_parallel,
    meta_head_only_info,
    index_column_index,
    index_column_type,
    input_file_name_col,
    is_dead_table: bool,
    pyarrow_schema: pa.Schema,
    use_hive: bool,
):
    (
        tot_rows_to_read,
        _,
        _,
        _,
        selected_cols,
        selected_cols_map,
        selected_partition_cols,
        partition_col_cat_dtypes,
        str_as_dict_cols,
        nullable_cols,
        sel_partition_names,
        partition_indices,
        sanitized_col_names,
        sel_partition_names_map,
    ) = pq_reader_params(
        meta_head_only_info,
        col_names,
        col_indices,
        partition_names,
        input_file_name_col,
        out_used_cols,
        index_column_index,
        index_column_type,
        out_types,
    )

    index_arr_ind = (
        selected_cols_map[index_column_index]
        if index_column_index is not None
        else None
    )
    index_arr_type = index_column_type
    py_table_type = TableType(tuple(out_types))
    if is_dead_table:
        py_table_type = types.none

    # table_idx is a list of index values for each array in the bodo.TableType being loaded from C++.
    # For a list column, the value is an integer which is the location of the column in the C++ Table.
    # Dead columns have the value -1.

    # For example if the Table Type is mapped like this: Table(arr0, arr1, arr2, arr3) and the
    # C++ representation is CPPTable(arr1, arr2), then table_idx = [-1, 0, 1, -1]

    # Note: By construction arrays will never be reordered (e.g. CPPTable(arr2, arr1)) in Iceberg
    # because we pass the col_names ordering.
    if is_dead_table:
        # If a table is dead we can skip the array for the table
        table_idx = None
    else:
        # index in cpp table for each column.
        # If a column isn't loaded we set the value to -1
        # and mark it as null in the conversion to Python
        table_idx = []
        j = 0
        input_file_name_col_idx = (
            col_indices[col_names.index(input_file_name_col)]
            if input_file_name_col is not None
            else None
        )
        for i, col_num in enumerate(col_indices):
            if j < len(out_used_cols) and i == out_used_cols[j]:
                col_idx = col_indices[i]
                if input_file_name_col_idx and col_idx == input_file_name_col_idx:
                    # input_file_name column goes at the end
                    table_idx.append(len(selected_cols) + len(sel_partition_names))
                elif col_idx in partition_indices:
                    c_name = sanitized_col_names[i]
                    table_idx.append(
                        len(selected_cols) + sel_partition_names_map[c_name]
                    )
                else:
                    table_idx.append(selected_cols_map[col_num])
                j += 1
            else:
                table_idx.append(-1)
        table_idx = np.array(table_idx, dtype=np.int64)

    pyarrow_schema_no_meta = pyarrow_schema.remove_metadata()

    comma = "," if extra_args else ""
    call_id = create_arg_hash(
        is_parallel,
        expr_filter_str,
        comma,
        storage_options,
        tot_rows_to_read,
        selected_cols,
        selected_cols_map,
        selected_partition_cols,
        partition_col_cat_dtypes,
        str_as_dict_cols,
        nullable_cols,
        sel_partition_names,
        partition_indices,
        sanitized_col_names,
        sel_partition_names_map,
        input_file_name_col,
        index_arr_ind,
        use_hive,
        is_dead_table,
        table_idx,
        index_arr_type,
        pyarrow_schema_no_meta,
        *extra_args,
    )
    func_text = f"def bodo_pq_reader_py(fname,{extra_args}):\n"
    # if it's an s3 url, get the region and pass it into the c++ code
    func_text += f"    ev = bodo.utils.tracing.Event('read_parquet', {is_parallel})\n"
    func_text += "    ev.add_attribute('g_fname', fname)\n"
    func_text += f'    _, filters = get_filters_pyobject("[]", "{expr_filter_str}", ({extra_args}{comma}))\n'
    # convert the filename, which could be a string or a list of strings, to a
    # PyObject to pass to C++. C++ just passes it through to parquet_pio.py::get_parquet_dataset()
    func_text += "    fname_py = get_fname_pyobject(fname)\n"

    # Add a dummy variable to the dict (empty dicts are not yet supported in Numba).
    storage_options["bodo_dummy"] = "dummy"
    func_text += f"    storage_options_py = get_storage_options_pyobject({str(storage_options)})\n"

    # Call pq_read_py_entry() in C++
    # single-element numpy array to return number of global rows from C++
    func_text += (
        f"    total_rows_np = np.array([0], dtype=np.int64)\n"
        f"    out_table = pq_read_py_entry(\n"
        f"        fname_py,\n"
        f"        {is_parallel},\n"
        f"        filters,\n"
        f"        storage_options_py,\n"
        f"        pyarrow_schema_{call_id},\n"
        f"        {tot_rows_to_read},\n"
        f"        selected_cols_arr_{call_id}.ctypes,\n"
        f"        {len(selected_cols)},\n"
        f"        nullable_cols_arr_{call_id}.ctypes,\n"
    )

    if len(selected_partition_cols) > 0:
        func_text += (
            f"        np.array({selected_partition_cols}, dtype=np.int32).ctypes,\n"
            f"        np.array({partition_col_cat_dtypes}, dtype=np.int32).ctypes,\n"
            f"        {len(selected_partition_cols)},\n"
        )
    else:
        func_text += "        0, 0, 0,\n"
    if len(str_as_dict_cols) > 0:
        # TODO pass array as global to function instead?
        func_text += f"        np.array({str_as_dict_cols}, dtype=np.int32).ctypes, {len(str_as_dict_cols)},\n"
    else:
        func_text += "        0, 0,\n"
    func_text += "        total_rows_np.ctypes,\n"
    # The C++ code only needs a flag
    func_text += f"        {input_file_name_col is not None},\n"
    func_text += f"        {use_hive},\n"
    func_text += "    )\n"
    func_text += "    check_and_propagate_cpp_exception()\n"

    func_text += "    total_rows = total_rows_np[0]\n"
    # Compute the number of rows that are stored in your chunk of the data.
    # This is necessary because we may avoid reading any columns but may not
    # be able to do the head only optimization.
    if is_parallel:
        func_text += "    local_rows = get_node_portion(total_rows, bodo.get_size(), bodo.get_rank())\n"
    else:
        func_text += "    local_rows = total_rows\n"

    # Extract the table and index from C++.
    if is_dead_table:
        func_text += "    T = None\n"
    else:
        func_text += f"    T = cpp_table_to_py_table(out_table, table_idx_{call_id}, py_table_type_{call_id}, 0)\n"
        if len(out_used_cols) == 0:
            # Set the table length using the total rows if don't load any columns
            func_text += "    T = set_table_len(T, local_rows)\n"
    if index_column_index is None:
        func_text += "    index_arr = None\n"
    else:
        func_text += f"    index_arr = array_from_cpp_table(out_table, {index_arr_ind}, index_arr_type)\n"
    func_text += "    delete_table(out_table)\n"
    func_text += "    ev.finalize()\n"
    func_text += "    return (total_rows, T, index_arr)\n"
    loc_vars = {}
    glbs = {
        f"py_table_type_{call_id}": py_table_type,
        f"table_idx_{call_id}": table_idx,
        f"selected_cols_arr_{call_id}": np.array(selected_cols, np.int32),
        f"nullable_cols_arr_{call_id}": np.array(nullable_cols, np.int32),
        f"pyarrow_schema_{call_id}": pyarrow_schema_no_meta,
        "index_arr_type": index_arr_type,
        "cpp_table_to_py_table": cpp_table_to_py_table,
        "array_from_cpp_table": array_from_cpp_table,
        "delete_table": delete_table,
        "check_and_propagate_cpp_exception": check_and_propagate_cpp_exception,
        "pq_read_py_entry": pq_read_py_entry,
        "get_filters_pyobject": get_filters_pyobject,
        "get_storage_options_pyobject": get_storage_options_pyobject,
        "get_fname_pyobject": get_fname_pyobject,
        "np": np,
        "pd": pd,
        "bodo": bodo,
        "get_node_portion": bodo.libs.distributed_api.get_node_portion,
        "set_table_len": bodo.hiframes.table.set_table_len,
    }

    pq_reader_py = bodo_exec(func_text, glbs, loc_vars, __name__)

    jit_func = numba.njit(pq_reader_py, no_cpython_wrapper=True, cache=True)
    return jit_func


def _gen_pq_reader_chunked_py(
    col_names: list[str],
    col_indices,
    out_used_cols,
    out_table_col_types: list[types.ArrayCompatible],
    storage_options,
    partition_names,
    expr_filter_str: str,
    extra_args,
    is_parallel,
    meta_head_only_info,
    index_column_index,
    index_column_type,
    input_file_name_col,
    is_dead_table: bool,
    pyarrow_schema: pa.Schema,
    use_hive: bool,
    chunksize: int,
    sql_op_id: int,
):
    """
    Generate Python code for streaming Parquet initialization impl.

    See _gen_pq_reader_py for base argument documentation

    Args:
        chunksize: Number of rows in each batch
        arrow_reader_t: Typing of ArrowReader output
    """

    call_id = next_label()
    comma = "," if extra_args else ""
    storage_options["bodo_dummy"] = "dummy"

    func_text = (
        f"def pq_reader_chunked_py(fname, {extra_args}):\n"
        f"    ev = bodo.utils.tracing.Event('read_parquet', {is_parallel})\n"
        f"    ev.add_attribute('g_fname', fname)\n"
        f'    _, filters = get_filters_pyobject("[]", "{expr_filter_str}", ({extra_args}{comma}))\n'
        f"    fname_py = get_fname_pyobject(fname)\n"
        # Add a dummy variable to the dict (empty dicts are not yet supported in numba).
        f"    storage_options_py = get_storage_options_pyobject({str(storage_options)})\n"
    )

    (
        tot_rows_to_read,
        _,
        _,
        _,
        selected_cols,
        _,
        selected_partition_cols,
        partition_col_cat_dtypes,
        str_as_dict_cols,
        nullable_cols,
        _,
        _,
        _,
        _,
    ) = pq_reader_params(
        meta_head_only_info,
        col_names,
        col_indices,
        partition_names,
        input_file_name_col,
        out_used_cols,
        index_column_index,
        index_column_type,
        out_table_col_types,
    )

    # Call pq_reader_init_py_entry() in C++
    # single-element numpy array to return number of global rows from C++
    func_text += "\n".join(
        [
            "    pq_reader = pq_reader_init_py_entry(",
            "        fname_py,",
            f"        {is_parallel},",
            "        filters,",
            "        storage_options_py,",
            f"        pyarrow_schema_{call_id},",
            f"        {tot_rows_to_read},",
            f"        selected_cols_arr_{call_id}.ctypes,",
            f"        {len(selected_cols)},",
            f"        nullable_cols_arr_{call_id}.ctypes,",
        ]
    )

    if len(selected_partition_cols) > 0:
        func_text += (
            f"        np.array({selected_partition_cols}, dtype=np.int32).ctypes,\n"
            f"        np.array({partition_col_cat_dtypes}, dtype=np.int32).ctypes,\n"
            f"        {len(selected_partition_cols)},\n"
        )
    else:
        func_text += "        0, 0, 0,\n"

    if len(str_as_dict_cols) > 0:
        # TODO pass array as global to function instead?
        func_text += f"        np.array({str_as_dict_cols}, dtype=np.int32).ctypes, {len(str_as_dict_cols)},\n"
    else:
        func_text += "        0, 0,\n"

    func_text += (
        # The C++ code only needs a flag
        f"        {input_file_name_col is not None},\n"
        f"        {chunksize},\n"
        f"        {use_hive},\n"
        f"        {sql_op_id},\n"
        f"        arrow_reader_t,\n"
        f"    )\n"
        f"    return pq_reader\n"
    )

    glbls = {
        f"selected_cols_arr_{call_id}": np.array(selected_cols, np.int32),
        f"nullable_cols_arr_{call_id}": np.array(nullable_cols, np.int32),
        f"pyarrow_schema_{call_id}": pyarrow_schema.remove_metadata(),
        "pq_reader_init_py_entry": pq_reader_init_py_entry,
        "get_filters_pyobject": get_filters_pyobject,
        "get_storage_options_pyobject": get_storage_options_pyobject,
        "get_fname_pyobject": get_fname_pyobject,
        "arrow_reader_t": ArrowReaderType(col_names, out_table_col_types),
        "np": np,
        "bodo": bodo,
    }

    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    pq_reader_py = loc_vars["pq_reader_chunked_py"]
    return numba.njit(pq_reader_py, no_cpython_wrapper=True)


@numba.njit
def get_fname_pyobject(fname):
    """Convert fname native object (which can be a string or a list of strings)
    to its corresponding PyObject by going through unboxing and boxing"""
    with bodo.no_warning_objmode(fname_py="read_parquet_fpath_type"):
        fname_py = fname
    return fname_py


numba.parfors.array_analysis.array_analysis_extensions[ParquetReader] = (
    bodo.ir.connector.connector_array_analysis
)
distributed_analysis.distributed_analysis_extensions[ParquetReader] = (
    bodo.ir.connector.connector_distributed_analysis
)
typeinfer.typeinfer_extensions[ParquetReader] = bodo.ir.connector.connector_typeinfer
ir_utils.visit_vars_extensions[ParquetReader] = bodo.ir.connector.visit_vars_connector
ir_utils.remove_dead_extensions[ParquetReader] = remove_dead_pq
numba.core.analysis.ir_extension_usedefs[ParquetReader] = (
    bodo.ir.connector.connector_usedefs
)
ir_utils.copy_propagate_extensions[ParquetReader] = (
    bodo.ir.connector.get_copies_connector
)
ir_utils.apply_copy_propagate_extensions[ParquetReader] = (
    bodo.ir.connector.apply_copies_connector
)
ir_utils.build_defs_extensions[ParquetReader] = (
    bodo.ir.connector.build_connector_definitions
)
remove_dead_column_extensions[ParquetReader] = pq_remove_dead_column
ir_extension_table_column_use[ParquetReader] = (
    bodo.ir.connector.connector_table_column_use
)
distributed_pass.distributed_run_extensions[ParquetReader] = pq_distributed_run


pq_read_py_entry = types.ExternalFunction(
    "pq_read_py_entry",
    table_type(
        read_parquet_fpath_type,  # path
        types.boolean,  # parallel
        parquet_predicate_type,  # filters
        storage_options_dict_type,  # storage_options
        pyarrow_schema_type,  # pyarrow_schema
        types.int64,  # tot_rows_to_read
        types.voidptr,  # _selected_fields
        types.int32,  # num_selected_fields
        types.voidptr,  # _is_nullable
        types.voidptr,  # selected_part_cols
        types.voidptr,  # part_cols_cat_dtype
        types.int32,  # num_partition_cols
        types.voidptr,  # str_as_dict_cols
        types.int32,  # num_str_as_dict_cols
        types.voidptr,  # total_rows_out
        types.boolean,  # input_file_name_col
        types.boolean,  # use_hive
    ),
)


@intrinsic
def pq_reader_init_py_entry(
    typingctx,
    path_t,
    parallel_t,
    filters_t,
    storage_options_t,
    pyarrow_schema_t,
    tot_rows_to_read_t,
    selected_fields_t,
    num_selected_fields,
    is_nullable_t,
    selected_part_cols_t,
    part_cols_cat_dtype_t,
    num_partition_cols_t,
    str_as_dict_cols_t,
    num_str_as_dict_cols_t,
    input_file_name_col_t,
    chunksize_t,
    use_hive_t,
    op_id_t,
    arrow_reader_t,
):  # pragma: no cover
    assert isinstance(arrow_reader_t, types.TypeRef) and isinstance(
        arrow_reader_t.instance_type, ArrowReaderType
    ), (
        "pq_reader_init_py_entry(): The last argument arrow_reader must by a TypeRef to an ArrowReader"
    )
    assert pyarrow_schema_t == pyarrow_schema_type, (
        "pq_reader_init_py_entry(): The 5th argument pyarrow_schema must by a PyArrow schema"
    )

    def codegen(context: "BaseContext", builder: "IRBuilder", signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),  # path void*
                lir.IntType(1),  # parallel bool
                lir.IntType(8).as_pointer(),  # filters PyObject*
                lir.IntType(8).as_pointer(),  # storage_options PyObject*
                lir.IntType(8).as_pointer(),  # pyarrow_schema PyObject*
                lir.IntType(64),  # tot_rows_to_read int64*
                lir.IntType(8).as_pointer(),  # _selected_fields void*
                lir.IntType(32),  # num_selected_fields int32
                lir.IntType(8).as_pointer(),  # _is_nullable void*
                lir.IntType(8).as_pointer(),  # selected_part_cols void*
                lir.IntType(8).as_pointer(),  # part_cols_cat_dtype void*
                lir.IntType(32),  # num_partition_cols int32
                lir.IntType(8).as_pointer(),  # str_as_dict_cols void*
                lir.IntType(32),  # num_str_as_dict_cols int32
                lir.IntType(1),  # input_file_name_col bool
                lir.IntType(64),  # batch_size int64
                lir.IntType(1),  # use_hive bool
                lir.IntType(64),  # op_id
            ],
        )

        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="pq_reader_init_py_entry"
        )

        pq_reader = builder.call(fn_tp, args[:-1])
        inlined_check_and_propagate_cpp_exception(context, builder)
        return pq_reader

    sig = arrow_reader_t.instance_type(
        read_parquet_fpath_type,  # path
        types.boolean,  # parallel
        parquet_predicate_type,  # filters
        storage_options_dict_type,  # storage_options
        pyarrow_schema_type,  # pyarrow_schema
        types.int64,  # tot_rows_to_read
        types.voidptr,  # _selected_fields
        types.int32,  # num_selected_fields
        types.voidptr,  # _is_nullable
        types.voidptr,  # selected_part_cols
        types.voidptr,  # part_cols_cat_dtype
        types.int32,  # num_partition_cols
        types.voidptr,  # str_as_dict_cols
        types.int32,  # num_str_as_dict_cols
        types.boolean,  # input_file_name_col
        types.int64,  # batch_size
        types.boolean,  # use_hive
        types.int64,  # op_id
        arrow_reader_t,  # typing only
    )
    return sig, codegen
