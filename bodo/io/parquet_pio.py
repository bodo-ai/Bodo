# Copyright (C) 2019 Bodo Inc. All rights reserved.
import os
import warnings
from collections import defaultdict
from glob import has_magic
from urllib.parse import urlparse

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
import pyarrow  # noqa
import pyarrow as pa  # noqa
import pyarrow.dataset as ds
from numba.core import ir, types
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
    box,
    intrinsic,
    models,
    overload,
    register_model,
    unbox,
)
from pyarrow._fs import PyFileSystem
from pyarrow.fs import FSSpecHandler

import bodo
import bodo.ir.parquet_ext
import bodo.utils.tracing as tracing
from bodo.hiframes.pd_categorical_ext import (
    CategoricalArrayType,
    PDCategoricalDtype,
)
from bodo.hiframes.table import TableType
from bodo.io.fs_io import (
    get_hdfs_fs,
    get_s3_fs_from_path,
    get_storage_options_pyobject,
    storage_options_dict_type,
)
from bodo.io.helpers import _get_numba_typ_from_pa_typ, is_nullable
from bodo.libs.array import (
    cpp_table_to_py_table,
    delete_table,
    info_from_table,
    info_to_array,
    table_type,
)
from bodo.libs.dict_arr_ext import dict_str_arr_type
from bodo.libs.distributed_api import get_end, get_start
from bodo.libs.str_ext import unicode_to_utf8
from bodo.transforms import distributed_pass
from bodo.utils.transform import get_const_value
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    FileInfo,
    get_overload_const_str,
)
from bodo.utils.utils import (
    check_and_propagate_cpp_exception,
    numba_to_c_type,
    sanitize_varname,
)

REMOTE_FILESYSTEMS = {"s3", "gcs", "gs", "http", "hdfs", "abfs", "abfss"}
# the ratio of total_uncompressed_size of a Parquet string column vs number of values,
# below which we read as dictionary-encoded string array
READ_STR_AS_DICT_THRESHOLD = 1.0

list_of_files_error_msg = ". Make sure the list/glob passed to read_parquet() only contains paths to files (no directories)"


class ParquetPredicateType(types.Type):
    """Type for predicate list for Parquet filtering (e.g. [["a", "==", 2]]).
    It is just a Python object passed as pointer to C++
    """

    def __init__(self):
        super(ParquetPredicateType, self).__init__(name="ParquetPredicateType()")


parquet_predicate_type = ParquetPredicateType()
types.parquet_predicate_type = parquet_predicate_type
register_model(ParquetPredicateType)(models.OpaqueModel)


@unbox(ParquetPredicateType)
def unbox_parquet_predicate_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return NativeValue(val)


@box(ParquetPredicateType)
def box_parquet_predicate_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return val


class ReadParquetFilepathType(types.Opaque):
    """Type for file path object passed to C++. It is just a Python object passed
    as a pointer to C++ (can be Python list of strings or Python string)
    """

    def __init__(self):
        super(ReadParquetFilepathType, self).__init__(name="ReadParquetFilepathType")


read_parquet_fpath_type = ReadParquetFilepathType()
types.read_parquet_fpath_type = read_parquet_fpath_type
register_model(ReadParquetFilepathType)(models.OpaqueModel)


@unbox(ReadParquetFilepathType)
def unbox_read_parquet_fpath_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return NativeValue(val)


class ParquetFileInfo(FileInfo):
    """FileInfo object passed to ForceLiteralArg for
    file name arguments that refer to a parquet dataset"""

    def __init__(
        self,
        columns,
        storage_options=None,
        input_file_name_col=None,
        read_as_dict_cols=None,
    ):
        self.columns = columns  # columns to select from parquet dataset
        self.storage_options = storage_options
        self.input_file_name_col = input_file_name_col
        self.read_as_dict_cols = read_as_dict_cols
        super().__init__()

    def _get_schema(self, fname):
        try:
            return parquet_file_schema(
                fname,
                selected_columns=self.columns,
                storage_options=self.storage_options,
                input_file_name_col=self.input_file_name_col,
                read_as_dict_cols=self.read_as_dict_cols,
            )
        except OSError as e:
            if "non-file path" in str(e):
                raise FileNotFoundError(str(e))
            raise


class ParquetHandler:
    """analyze and transform parquet IO calls"""

    def __init__(self, func_ir, typingctx, args, _locals):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.args = args
        self.locals = _locals

    def gen_parquet_read(
        self,
        file_name,
        lhs,
        columns,
        storage_options=None,
        input_file_name_col=None,
        read_as_dict_cols=None,
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
                ),
            )

            got_schema = False
            # get_const_value forces variable to be literal which should convert it to
            # FilenameType. If so, the schema will be part of the type
            var_def = guard(get_definition, self.func_ir, file_name)
            if isinstance(var_def, ir.Arg):
                typ = self.args[var_def.index]
                if isinstance(typ, types.FilenameType):
                    (
                        col_names,
                        col_types,
                        index_col,
                        col_indices,
                        partition_names,
                        unsupported_columns,
                        unsupported_arrow_types,
                    ) = typ.schema
                    got_schema = True
            if not got_schema:
                (
                    col_names,
                    col_types,
                    index_col,
                    col_indices,
                    partition_names,
                    unsupported_columns,
                    unsupported_arrow_types,
                ) = parquet_file_schema(
                    file_name_str,
                    columns,
                    storage_options=storage_options,
                    input_file_name_col=input_file_name_col,
                    read_as_dict_cols=read_as_dict_cols,
                )
        else:
            col_names_total = list(table_types.keys())
            # Create a map for efficient index lookup
            col_names_total_map = {c: i for i, c in enumerate(col_names_total)}
            col_types_total = [t for t in table_types.values()]
            index_col = "index" if "index" in col_names_total_map else None
            # TODO: allow specifying types of only selected columns
            if columns is None:
                selected_columns = col_names_total
            else:
                selected_columns = columns
            col_indices = [col_names_total_map[c] for c in selected_columns]
            col_types = [
                col_types_total[col_names_total_map[c]] for c in selected_columns
            ]
            col_names = selected_columns
            index_col = index_col if index_col in col_names else None
            partition_names = []
            # If a user provides the schema, all types must be valid Bodo types.
            unsupported_columns = []
            unsupported_arrow_types = []

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

        data_arrs = [
            ir.Var(scope, mk_unique_var("pq_table"), loc),
            ir.Var(scope, mk_unique_var("pq_index"), loc),
        ]

        nodes = [
            bodo.ir.parquet_ext.ParquetReader(
                file_name,
                lhs.name,
                col_names,
                col_indices,
                col_types,
                data_arrs,
                loc,
                partition_names,
                storage_options,
                index_column_index,
                index_column_type,
                input_file_name_col,
                unsupported_columns,
                unsupported_arrow_types,
            )
        ]

        return col_names, data_arrs, index_col, nodes, col_types, index_column_type


def pq_distributed_run(
    pq_node,
    array_dists,
    typemap,
    calltypes,
    typingctx,
    targetctx,
    meta_head_only_info=None,
):
    """lower ParquetReader into regular Numba nodes. Generates code for Parquet
    data read.
    """
    n_cols = len(pq_node.out_vars)
    dnf_filter_str = "None"
    expr_filter_str = "None"

    filter_map, filter_vars = bodo.ir.connector.generate_filter_map(pq_node.filters)
    extra_args = ", ".join(filter_map.values())
    dnf_filter_str, expr_filter_str = bodo.ir.connector.generate_arrow_filters(
        pq_node.filters,
        filter_map,
        filter_vars,
        pq_node.original_df_colnames,
        pq_node.partition_names,
        pq_node.original_out_types,
        typemap,
        "parquet",
        output_dnf=False,
    )
    arg_names = ", ".join(f"out{i}" for i in range(n_cols))
    func_text = f"def pq_impl(fname, {extra_args}):\n"
    # total_rows is used for setting total size variable below
    func_text += (
        f"    (total_rows, {arg_names},) = _pq_reader_py(fname, {extra_args})\n"
    )

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    pq_impl = loc_vars["pq_impl"]

    # Add debug info about column pruning and dictionary encoded arrays.
    if bodo.user_logging.get_verbose_level() >= 1:
        # State which columns are pruned
        pq_source = pq_node.loc.strformat()
        pq_cols = []
        dict_encoded_cols = []
        for i in pq_node.out_used_cols:
            colname = pq_node.df_colnames[i]
            pq_cols.append(colname)
            if isinstance(
                pq_node.out_types[i], bodo.libs.dict_arr_ext.DictionaryArrayType
            ):
                dict_encoded_cols.append(colname)
        pruning_msg = (
            "Finish column pruning on read_parquet node:\n%s\nColumns loaded %s\n"
        )
        bodo.user_logging.log_message(
            "Column Pruning",
            pruning_msg,
            pq_source,
            pq_cols,
        )
        # Log if any columns use dictionary encoded arrays.
        if dict_encoded_cols:
            encoding_msg = "Finished optimized encoding on read_parquet node:\n%s\nColumns %s using dictionary encoding to reduce memory usage.\n"
            bodo.user_logging.log_message(
                "Dictionary Encoding",
                encoding_msg,
                pq_source,
                dict_encoded_cols,
            )

    # parallel read flag
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
                f"pandas.read_parquet(): 1 or more columns found with Arrow types that are not supported in Bodo and could not be eliminated. "
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
                    f"Column '{pq_node.df_colnames[col_num]}' with unsupported arrow type {pq_node.unsupported_arrow_types[idx]}"
                )
                idx += 1
            total_msg = "\n".join(msg_list)
            raise BodoError(total_msg, loc=pq_node.loc)

    pq_reader_py = _gen_pq_reader_py(
        pq_node.df_colnames,
        pq_node.col_indices,
        pq_node.out_used_cols,
        pq_node.out_types,
        pq_node.storage_options,
        pq_node.partition_names,
        dnf_filter_str,
        expr_filter_str,
        extra_args,
        parallel,
        meta_head_only_info,
        pq_node.index_column_index,
        pq_node.index_column_type,
        pq_node.input_file_name_col,
        not pq_node.is_live_table,
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

    # assign output table
    nodes[-2].target = pq_node.out_vars[0]
    # assign output index array
    nodes[-1].target = pq_node.out_vars[1]
    # At most one of the table and the index
    # can be dead because otherwise the whole
    # node should have already been removed.
    assert not (
        pq_node.index_column_index is None and not pq_node.is_live_table
    ), "At most one of table and index should be dead if the Parquet IR node is live"
    if pq_node.index_column_index is None:
        # If the index_col is dead, remove the node.
        nodes.pop(-1)
    elif not pq_node.is_live_table:
        # If the table is dead, remove the node
        nodes.pop(-2)

    return nodes


distributed_pass.distributed_run_extensions[
    bodo.ir.parquet_ext.ParquetReader
] = pq_distributed_run


def get_filters_pyobject(dnf_filter_str, expr_filter_str, vars):  # pragma: no cover
    pass


@overload(get_filters_pyobject, no_unliteral=True)
def overload_get_filters_pyobject(dnf_filter_str, expr_filter_str, var_tup):
    """generate a pyobject for filter expression to pass to C++"""
    dnf_filter_str_val = get_overload_const_str(dnf_filter_str)
    expr_filter_str_val = get_overload_const_str(expr_filter_str)
    var_unpack = ", ".join(f"f{i}" for i in range(len(var_tup)))
    func_text = "def impl(dnf_filter_str, expr_filter_str, var_tup):\n"
    if len(var_tup):
        func_text += f"  {var_unpack}, = var_tup\n"
    func_text += "  with numba.objmode(dnf_filters_py='parquet_predicate_type', expr_filters_py='parquet_predicate_type'):\n"
    func_text += f"    dnf_filters_py = {dnf_filter_str_val}\n"
    func_text += f"    expr_filters_py = {expr_filter_str_val}\n"
    func_text += "  return (dnf_filters_py, expr_filters_py)\n"
    loc_vars = {}
    exec(func_text, globals(), loc_vars)
    return loc_vars["impl"]


@numba.njit
def get_fname_pyobject(fname):
    """Convert fname native object (which can be a string or a list of strings)
    to its corresponding PyObject by going through unboxing and boxing"""
    with numba.objmode(fname_py="read_parquet_fpath_type"):
        fname_py = fname
    return fname_py


def _gen_pq_reader_py(
    col_names,
    col_indices,
    out_used_cols,
    out_types,
    storage_options,
    partition_names,
    dnf_filter_str,
    expr_filter_str,
    extra_args,
    is_parallel,
    meta_head_only_info,
    index_column_index,
    index_column_type,
    input_file_name_col,
    is_dead_table,
):

    # a unique int used to create global variables with unique names
    call_id = next_label()

    comma = "," if extra_args else ""
    func_text = f"def pq_reader_py(fname,{extra_args}):\n"
    # if it's an s3 url, get the region and pass it into the c++ code
    func_text += f"    ev = bodo.utils.tracing.Event('read_parquet', {is_parallel})\n"
    func_text += f"    ev.add_attribute('g_fname', fname)\n"
    func_text += f'    dnf_filters, expr_filters = get_filters_pyobject("{dnf_filter_str}", "{expr_filter_str}", ({extra_args}{comma}))\n'
    # convert the filename, which could be a string or a list of strings, to a
    # PyObject to pass to C++. C++ just passes it through to parquet_pio.py::get_parquet_dataset()
    func_text += "    fname_py = get_fname_pyobject(fname)\n"

    # Add a dummy variable to the dict (empty dicts are not yet supported in numba).
    storage_options["bodo_dummy"] = "dummy"
    func_text += f"    storage_options_py = get_storage_options_pyobject({str(storage_options)})\n"

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
    input_file_name_col = (
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
        int(is_nullable(out_types[col_indices_map[col_in_idx]]))
        if col_in_idx != index_column_index
        else int(is_nullable(index_column_type))
        for col_in_idx in selected_cols
    ]

    # XXX is special handling needed for table format?
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

    # Call pq_read() in C++
    # single-element numpy array to return number of global rows from C++
    func_text += f"    total_rows_np = np.array([0], dtype=np.int64)\n"
    func_text += f"    out_table = pq_read(\n"
    func_text += f"        fname_py, {is_parallel},\n"
    func_text += f"        dnf_filters, expr_filters,\n"
    func_text += f"        storage_options_py, {tot_rows_to_read}, selected_cols_arr_{call_id}.ctypes,\n"
    func_text += f"        {len(selected_cols)},\n"
    func_text += f"        nullable_cols_arr_{call_id}.ctypes,\n"
    if len(selected_partition_cols) > 0:
        func_text += (
            f"        np.array({selected_partition_cols}, dtype=np.int32).ctypes,\n"
        )
        func_text += (
            f"        np.array({partition_col_cat_dtypes}, dtype=np.int32).ctypes,\n"
        )
        func_text += f"        {len(selected_partition_cols)},\n"
    else:
        func_text += f"        0, 0, 0,\n"
    if len(str_as_dict_cols) > 0:
        # TODO pass array as global to function instead?
        func_text += f"        np.array({str_as_dict_cols}, dtype=np.int32).ctypes, {len(str_as_dict_cols)},\n"
    else:
        func_text += f"        0, 0,\n"
    func_text += f"        total_rows_np.ctypes,\n"
    # The C++ code only needs a flag
    func_text += f"        {input_file_name_col is not None},\n"
    func_text += f"    )\n"
    func_text += f"    check_and_propagate_cpp_exception()\n"

    func_text += f"    total_rows = total_rows_np[0]\n"
    # Compute the number of rows that are stored in your chunk of the data.
    # This is necessary because we may avoid reading any columns but may not
    # be able to do the head only optimization.
    if is_parallel:
        func_text += f"    local_rows = get_node_portion(total_rows, bodo.get_size(), bodo.get_rank())\n"
    else:
        func_text += f"    local_rows = total_rows\n"

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

    # Extract the table and index from C++.
    if is_dead_table:
        func_text += "    T = None\n"
    else:
        func_text += f"    T = cpp_table_to_py_table(out_table, table_idx_{call_id}, py_table_type_{call_id})\n"
        # Set the table length
        func_text += f"    T = set_table_len(T, local_rows)\n"
    if index_column_index is None:
        func_text += "    index_arr = None\n"
    else:
        index_arr_ind = selected_cols_map[index_column_index]
        func_text += f"    index_arr = info_to_array(info_from_table(out_table, {index_arr_ind}), index_arr_type)\n"
    func_text += f"    delete_table(out_table)\n"
    func_text += f"    ev.finalize()\n"
    func_text += f"    return (total_rows, T, index_arr)\n"
    loc_vars = {}
    glbs = {
        f"py_table_type_{call_id}": py_table_type,
        f"table_idx_{call_id}": table_idx,
        f"selected_cols_arr_{call_id}": np.array(selected_cols, np.int32),
        f"nullable_cols_arr_{call_id}": np.array(nullable_cols, np.int32),
        "index_arr_type": index_arr_type,
        "cpp_table_to_py_table": cpp_table_to_py_table,
        "info_to_array": info_to_array,
        "info_from_table": info_from_table,
        "delete_table": delete_table,
        "check_and_propagate_cpp_exception": check_and_propagate_cpp_exception,
        "pq_read": _pq_read,
        "unicode_to_utf8": unicode_to_utf8,
        "get_filters_pyobject": get_filters_pyobject,
        "get_storage_options_pyobject": get_storage_options_pyobject,
        "get_fname_pyobject": get_fname_pyobject,
        "np": np,
        "pd": pd,
        "bodo": bodo,
        "get_node_portion": bodo.libs.distributed_api.get_node_portion,
        "set_table_len": bodo.hiframes.table.set_table_len,
    }

    exec(func_text, glbs, loc_vars)
    pq_reader_py = loc_vars["pq_reader_py"]

    jit_func = numba.njit(pq_reader_py, no_cpython_wrapper=True)
    return jit_func


def unify_schemas(schemas):
    """
    Same as pyarrow.unify_schemas with the difference
    that we unify `large_string` and `string` types to `string`,
    (Arrow considers them incompatible).
    Note that large_strings is not a property of parquet, but
    rather a decision made by Arrow on how to store string data
    in memory. For Bodo, we can have Arrow always read as regular
    strings and convert to Bodo's representation during read.
    Similarly, we also unify `large_binary` and `binary` to `binary`.
    We also convert LargeListType to regular list type (the type inside
    is not modified).
    Additionally, pa.list_(pa.large_string()) is converted to
    pa.list_(pa.string()).
    """
    # first replace large_string with string in every schema
    new_schemas = []
    for schema in schemas:
        for i in range(len(schema)):
            f = schema.field(i)
            if f.type == pa.large_string():
                schema = schema.set(i, f.with_type(pa.string()))
            elif f.type == pa.large_binary():
                schema = schema.set(i, f.with_type(pa.binary()))
            elif isinstance(
                f.type, (pa.ListType, pa.LargeListType)
            ) and f.type.value_type in (pa.string(), pa.large_string()):
                # This handles the pa.list_(pa.large_string()) case
                # that the next `elif` doesn't.
                schema = schema.set(
                    i,
                    f.with_type(
                        # We want to retain the name (e.g. 'element'), so we pass
                        # in a field to pa.list_ instead of a simple string type
                        # which would use 'item' by default.
                        pa.list_(pa.field(f.type.value_field.name, pa.string()))
                    ),
                )
            elif isinstance(f.type, pa.LargeListType):
                schema = schema.set(
                    i,
                    f.with_type(
                        # We want to retain the name (e.g. 'element'), so we pass
                        # in a field to pa.list_ instead of a simple string type
                        # which would use 'item' by default.
                        pa.list_(pa.field(f.type.value_field.name, f.type.value_type))
                    ),
                )
            # TODO handle arbitrary nested types
        new_schemas.append(schema)
    # now we run Arrow's regular schema unification
    return pa.unify_schemas(new_schemas)


class ParquetDataset(object):
    """Stores information about parquet dataset that is needed at compile time
    and runtime (to read the dataset). Stores the list of fragments
    (pieces) that form the dataset and filesystem object to read them.
    All of this is obtained at rank 0 using Arrow's pq.ParquetDataset() API
    (ParquetDatasetV2) and this object is broadcasted to all ranks.
    """

    def __init__(self, pa_pq_dataset, prefix=""):
        self.schema = pa_pq_dataset.schema  # Arrow schema
        # We don't store the filesystem initially, and instead set it after
        # ParquetDataset is broadcasted. This is because some filesystems
        # might have pickling issues, and also because of the extra cost of
        # creating the filesystem during unpickling. Instead, all ranks
        # initialize the filesystem in parallel at the same time and only once
        self.filesystem = None
        # total number of rows in the dataset (after applying filters). This
        # is computed at runtime in `get_parquet_dataset`
        self._bodo_total_rows = 0
        # prefix that needs to be added to paths of parquet pieces to get the
        # full path to the file
        self._prefix = prefix
        # XXX pa_pq_dataset.partitioning can't be pickled, so we reconstruct
        # manually after broadcasting the dataset (see __setstate__ below)
        self.partitioning = None
        partitioning = pa_pq_dataset.partitioning
        # For some datasets, partitioning.schema contains the
        # full schema of the dataset when there aren't any partition columns
        # (bug in Arrow?) so to know if there are partition columns we also
        # need to check that the partitioning schema is not equal to the
        # full dataset schema.
        # XXX is there a better way to get partition column names?
        self.partition_names = (
            []
            if partitioning is None or partitioning.schema == pa_pq_dataset.schema
            else list(partitioning.schema.names)
        )
        # partitioning_dictionaries is an Arrow array containing the
        # partition values
        if self.partition_names:
            self.partitioning_dictionaries = partitioning.dictionaries
            self.partitioning_cls = partitioning.__class__
            self.partitioning_schema = partitioning.schema
        else:
            self.partitioning_dictionaries = {}
        # Convert large_string Arrow types to string
        # (see comment in bodo.io.parquet_pio.unify_schemas)
        for i in range(len(self.schema)):
            f = self.schema.field(i)
            if f.type == pa.large_string():
                self.schema = self.schema.set(i, f.with_type(pa.string()))
        # IMPORTANT: only include partition columns in filters passed to
        # pq.ParquetDataset(), otherwise `get_fragments` could look inside the
        # parquet files
        self.pieces = [
            ParquetPiece(frag, partitioning, self.partition_names)
            for frag in pa_pq_dataset._dataset.get_fragments(
                filter=pa_pq_dataset._filter_expression
            )
        ]

    def set_fs(self, fs):
        """Set filesystem (to read fragments)"""
        self.filesystem = fs
        for p in self.pieces:
            p.filesystem = fs

    def __setstate__(self, state):
        """called when unpickling"""
        self.__dict__ = state
        if self.partition_names:
            # We do this because there is an error (bug?) when pickling
            # Arrow HivePartitioning objects
            part_dicts = {
                p: self.partitioning_dictionaries[i]
                for i, p in enumerate(self.partition_names)
            }
            self.partitioning = self.partitioning_cls(
                self.partitioning_schema, part_dicts
            )


class ParquetPiece(object):
    """Parquet dataset piece (file) information and Arrow objects to query
    metadata"""

    def __init__(self, frag, partitioning, partition_names):
        # We don't store the frag initially because we broadcast the dataset from rank 0,
        # and PyArrow has issues (un)pickling the frag, because it opens the file to access
        # metadata when pickling and/or unpickling. This can cause a massive slowdown
        # because a single process will try opening all the files in the dataset. Arrow 8
        # solved the issue when pickling, but I still see it when unpickling, reason
        # unknown (needs investigation). To check, simply pickle and unpickle the
        # bodo.io.parquet_io.ParquetDataset object on rank 0
        self._frag = None
        # needed to open the fragment (see frag property below)
        self.format = frag.format
        self.path = frag.path
        # number of rows in this piece after applying filters. This
        # is computed at runtime in `get_parquet_dataset`
        self._bodo_num_rows = 0
        self.partition_keys = []
        if partitioning is not None:
            # XXX these are not ordered by partitions or in inverse order for some reason
            self.partition_keys = ds._get_partition_keys(frag.partition_expression)
            self.partition_keys = [
                (
                    part_name,
                    partitioning.dictionaries[i]
                    .index(self.partition_keys[part_name])
                    .as_py(),
                )
                for i, part_name in enumerate(partition_names)
            ]

    @property
    def frag(self):
        """returns the Arrow ParquetFileFragment associated with this piece"""
        if self._frag is None:
            self._frag = self.format.make_fragment(
                self.path,
                self.filesystem,
            )
            del self.format
        return self._frag

    @property
    def metadata(self):
        """returns the Arrow metadata of this piece"""
        return self.frag.metadata

    @property
    def num_row_groups(self):
        """returns the number of row groups in this piece"""
        return self.frag.num_row_groups


def get_parquet_dataset(
    fpath,
    get_row_counts=True,
    dnf_filters=None,
    expr_filters=None,
    storage_options=None,
    read_categories=False,
    is_parallel=False,  # only used with get_row_counts=True
    tot_rows_to_read=None,
    typing_pa_schema=None,
    partitioning="hive",
):
    """get ParquetDataset object for 'fpath' and set the number of total rows as an
    attribute. Also, sets the number of rows per file as an attribute of
    ParquetDatasetPiece objects.
    'filters' are used for predicate pushdown which prunes the unnecessary pieces.
    'read_categories': read categories of DictionaryArray and store in returned dataset
    object, used during typing.
    'get_row_counts': This is only true at runtime, and indicates that we need
    to get the number of rows of each piece in the parquet dataset.
    'is_parallel' : True if reading in parallel
    'tot_rows_to_read' : total number of rows to read from dataset. Used at runtime
    for example if doing df.head(tot_rows_to_read) where df is the output of
    read_parquet()
    'typing_pa_schema': PyArrow schema determined at compile time. When provided,
    we should validate that the unified schema of all files matches this schema,
    and throw an error otherwise. Currently this is only used in the case
    of iceberg, but should be expanded to all use cases.
    https://bodo.atlassian.net/browse/BE-2787
    """

    # NOTE: This function obtains the metadata for a parquet dataset and works
    # in the same way regardless of whether the read is going to be parallel or
    # replicated. In all cases rank 0 will get the ParquetDataset from pyarrow,
    # broadcast it to all ranks, and they will divide the work of getting the
    # number of rows in each file of the dataset

    # get_parquet_dataset can be called both at both compile and run time. We
    # only want to trace it at run time, so take advantage of get_row_counts flag
    # to know if this is runtime
    if get_row_counts:
        ev = tracing.Event("get_parquet_dataset")

    import time

    import pyarrow as pa
    import pyarrow.parquet as pq
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    if isinstance(fpath, list):
        # list of file paths
        parsed_url = urlparse(fpath[0])
        protocol = parsed_url.scheme
        bucket_name = parsed_url.netloc  # netloc can be empty string (e.g. non s3)
        for i in range(len(fpath)):
            f = fpath[i]
            u_p = urlparse(f)
            # make sure protocol and bucket name of every file matches
            if u_p.scheme != protocol:
                raise BodoError(
                    "All parquet files must use the same filesystem protocol"
                )
            if u_p.netloc != bucket_name:
                raise BodoError("All parquet files must be in the same S3 bucket")
            fpath[i] = f.rstrip("/")
    else:
        parsed_url = urlparse(fpath)
        protocol = parsed_url.scheme
        fpath = fpath.rstrip("/")

    if protocol in {"gcs", "gs"}:
        try:
            import gcsfs
        except ImportError:
            message = (
                "Couldn't import gcsfs, which is required for Google cloud access."
                " gcsfs can be installed by calling"
                " 'conda install -c conda-forge gcsfs'.\n"
            )
            raise BodoError(message)

    if protocol == "http":
        try:
            import fsspec
        except ImportError:
            message = (
                "Couldn't import fsspec, which is required for http access."
                " fsspec can be installed by calling"
                " 'conda install -c conda-forge fsspec'.\n"
            )

    fs = []

    def getfs(parallel=False):
        # NOTE: add remote filesystems to REMOTE_FILESYSTEMS
        if len(fs) == 1:
            return fs[0]
        if protocol == "s3":
            fs.append(
                get_s3_fs_from_path(
                    fpath, parallel=parallel, storage_options=storage_options
                )
                if not isinstance(fpath, list)
                else get_s3_fs_from_path(
                    fpath[0], parallel=parallel, storage_options=storage_options
                )
            )
        elif protocol in {"gcs", "gs"}:
            # TODO pass storage_options to GCSFileSystem
            google_fs = gcsfs.GCSFileSystem(token=None)
            fs.append(PyFileSystem(FSSpecHandler(google_fs)))
        elif protocol == "http":
            fs.append(PyFileSystem(FSSpecHandler(fsspec.filesystem("http"))))
        elif protocol in {"hdfs", "abfs", "abfss"}:  # pragma: no cover
            fs.append(
                get_hdfs_fs(fpath)
                if not isinstance(fpath, list)
                else get_hdfs_fs(fpath[0])
            )
        else:
            fs.append(pa.fs.LocalFileSystem())
        return fs[0]

    def glob(protocol, fs, path):
        """Return a possibly-empty list of path names that match glob pattern
        given by path"""
        if not protocol and fs is None:
            from fsspec.implementations.local import LocalFileSystem

            fs = LocalFileSystem()
        if isinstance(fs, pa.fs.FileSystem):
            from fsspec.implementations.arrow import ArrowFSWrapper

            fs = ArrowFSWrapper(fs)
        try:
            files = fs.glob(path)
        except:  # pragma: no cover
            raise BodoError(f"glob pattern expansion not supported for {protocol}")
        if len(files) == 0:
            raise BodoError("No files found matching glob pattern")
        return files

    validate_schema = False
    if get_row_counts:
        # Getting row counts and schema validation is going to be
        # distributed across ranks, so every rank will need a filesystem
        # object to query the metadata of their assigned pieces.
        # We have seen issues in the past with broadcasting some filesystem
        # objects (e.g. s3) and even if they don't exist in Arrow >= 8
        # broadcasting the filesystem adds extra time, so instead we initialize
        # the filesystem before the broadcast. That way all ranks do it in parallel
        # at the same time.
        _ = getfs(parallel=True)
        validate_schema = bodo.parquet_validate_schema

    if bodo.get_rank() == 0:
        nthreads = 1  # number of threads to use on rank 0 to collect metadata
        cpu_count = os.cpu_count()
        if cpu_count is not None and cpu_count > 1:
            nthreads = cpu_count // 2
        try:
            if get_row_counts:
                ev_pq_ds = tracing.Event("pq.ParquetDataset", is_parallel=False)
                if tracing.is_tracing():
                    # only do the work of converting dnf_filters to string
                    # if tracing is enabled
                    ev_pq_ds.add_attribute("g_dnf_filter", str(dnf_filters))
            pa_default_io_thread_count = pa.io_thread_count()
            pa.set_io_thread_count(nthreads)

            prefix = ""
            if protocol == "s3":
                prefix = "s3://"
            elif protocol in {"hdfs", "abfs", "abfss"}:
                # HDFS filesystem is initialized with host:port info. Once
                # initialized, the filesystem needs the <protocol>://<host><port>
                # prefix removed to query and access files
                prefix = f"{protocol}://{parsed_url.netloc}"
            if prefix:
                if isinstance(fpath, list):
                    fpath_noprefix = [f[len(prefix) :] for f in fpath]
                else:
                    fpath_noprefix = fpath[len(prefix) :]
            else:
                fpath_noprefix = fpath

            if isinstance(fpath_noprefix, list):
                # Expand any glob strings in the list in order to generate a
                # single list of fully realized paths to parquet files.
                # For example: ["A/a.pq", "B/*.pq"] might expand to
                # ["A/a.pq", "B/part-0.pq", "B/part-1.pq"]
                new_fpath = []
                for p in fpath_noprefix:
                    if has_magic(p):
                        new_fpath += glob(protocol, getfs(), p)
                    else:
                        new_fpath.append(p)
                fpath_noprefix = new_fpath
            elif has_magic(fpath_noprefix):
                fpath_noprefix = glob(protocol, getfs(), fpath_noprefix)

            dataset = pq.ParquetDataset(
                fpath_noprefix,
                filesystem=getfs(),
                filters=None,  # we pass filters manually below because of Arrow bug
                use_legacy_dataset=False,  # Use ParquetDatasetV2
                partitioning=partitioning,
            )
            if dnf_filters is not None:
                # XXX This is actually done inside _ParquetDatasetV2 constructor,
                # but a bug in Arrow prevents passing compute expression filters.
                # We can remove this and pass the filters to ParquetDataset
                # constructor once the bug is fixed.
                dataset._filters = dnf_filters
                dataset._filter_expression = pq._filters_to_expression(dnf_filters)

            num_files_before_filter = len(dataset.files)
            # If there are dnf_filters files are filtered in ParquetDataset constructor
            dataset = ParquetDataset(dataset, prefix)
            # restore pyarrow default IO thread count
            pa.set_io_thread_count(pa_default_io_thread_count)

            # If typing schema is available, then use that as the baseline
            # schema to unify with, else get it from the dataset.
            # This is important for getting understandable errors in cases where
            # files have different schemas, some of which may or may not match
            # the iceberg schema. `get_dataset_schema` essentially gets the
            # schema of the first file. So, starting with that schema will
            # raise errors such as
            # "pyarrow.lib.ArrowInvalid: No match for FieldRef.Name(TY)"
            # where TY is a column that originally had a different name.
            # Therefore, it's better to start with the expected schema,
            # and then raise the errors correctly after validation.
            if typing_pa_schema:
                # NOTE: typing_pa_schema must include partitions
                dataset.schema = typing_pa_schema

            if get_row_counts:
                if dnf_filters is not None:
                    ev_pq_ds.add_attribute(
                        "num_pieces_before_filter", num_files_before_filter
                    )
                    ev_pq_ds.add_attribute(
                        "num_pieces_after_filter", len(dataset.pieces)
                    )
                ev_pq_ds.finalize()
        except Exception as e:
            # See note in s3_list_dir_fnames
            # In some cases, OSError/FileNotFoundError can propagate back to numba and comes back as InternalError.
            # where numba errors are hidden from the user.
            # See [BE-1188] for an example
            # Raising a BodoError lets message comes back and seen by the user.
            if isinstance(e, IsADirectoryError):
                # We supress Arrow's error message since it doesn't apply to Bodo
                # (the bit about doing a union of datasets)
                e = BodoError(list_of_files_error_msg)
            elif isinstance(fpath, list) and isinstance(
                e, (OSError, FileNotFoundError)
            ):
                e = BodoError(str(e) + list_of_files_error_msg)
            else:
                e = BodoError(f"error from pyarrow: {type(e).__name__}: {str(e)}\n")
            comm.bcast(e)
            raise e

        if get_row_counts:
            ev_bcast = tracing.Event("bcast dataset")
        dataset = comm.bcast(dataset)
    else:
        if get_row_counts:
            ev_bcast = tracing.Event("bcast dataset")
        dataset = comm.bcast(None)
        if isinstance(dataset, Exception):  # pragma: no cover
            error = dataset
            raise error

    # As mentioned above, we don't want to broadcast the filesystem because it
    # adds time (so initially we didn't include it in the dataset). We add
    # it to the dataset now that it's been broadcasted
    dataset.set_fs(getfs())

    if get_row_counts:
        ev_bcast.finalize()

    if get_row_counts and tot_rows_to_read == 0:
        get_row_counts = validate_schema = False
    if get_row_counts or validate_schema:
        # getting row counts and validating schema requires reading
        # the file metadata from the parquet files and is very expensive
        # for datasets consisting of many files, so we do this in parallel
        if get_row_counts and tracing.is_tracing():
            ev_row_counts = tracing.Event("get_row_counts")
            ev_row_counts.add_attribute("g_num_pieces", len(dataset.pieces))
            ev_row_counts.add_attribute("g_expr_filters", str(expr_filters))
        ds_scan_time = 0.0
        num_pieces = len(dataset.pieces)
        start = get_start(num_pieces, bodo.get_size(), bodo.get_rank())
        end = get_end(num_pieces, bodo.get_size(), bodo.get_rank())
        total_rows_chunk = 0
        total_row_groups_chunk = 0
        total_row_groups_size_chunk = 0
        valid = True  # True if schema of all parquet files match
        if expr_filters is not None:
            import random

            random.seed(37)
            pieces = random.sample(dataset.pieces, k=len(dataset.pieces))
        else:
            pieces = dataset.pieces

        fpaths = [p.path for p in pieces[start:end]]
        # Presumably the work is partitioned more or less equally among ranks,
        # and we are mostly (or just) reading metadata, so we assign four IO
        # threads to every rank
        nthreads = min(int(os.environ.get("BODO_MIN_IO_THREADS", 4)), 4)
        pa.set_io_thread_count(nthreads)
        pa.set_cpu_count(nthreads)
        # Use dataset scanner API to get exact row counts when
        # filter is applied. Arrow will try to calculate this by
        # by reading only the file's metadata, and if it needs to
        # access data it will read as less as possible (only the
        # required columns and only subset of row groups if possible)
        error = None
        try:
            dataset_ = ds.dataset(
                fpaths,
                filesystem=dataset.filesystem,
                partitioning=dataset.partitioning,
            )
            for piece, frag in zip(pieces[start:end], dataset_.get_fragments()):

                # The validation (and unification) step needs to happen before the
                # scan on the fragment since otherwise it will fail in case the
                # file schema doesn't match the dataset schema exactly.
                # Currently this is only applicable for Iceberg reads.
                if validate_schema:
                    # Two files are compatible if arrow can unify their schemas.
                    file_schema = frag.metadata.schema.to_arrow_schema()
                    fileset_schema_names = set(file_schema.names)
                    # Check the names are the same because pa.unify_schemas
                    # will unify a schema where a column is in 1 file but not
                    # another.
                    dataset_schema_names = set(dataset.schema.names) - set(
                        dataset.partition_names
                    )
                    if dataset_schema_names != fileset_schema_names:
                        added_columns = fileset_schema_names - dataset_schema_names
                        missing_columns = dataset_schema_names - fileset_schema_names
                        msg = f"Schema in {piece} was different.\n"
                        if added_columns:
                            msg += f"File contains column(s) {added_columns} not found in other files in the dataset.\n"
                        if missing_columns:
                            msg += f"File missing column(s) {missing_columns} found in other files in the dataset.\n"
                        raise BodoError(msg)
                    try:
                        dataset.schema = unify_schemas([dataset.schema, file_schema])
                    except Exception as e:
                        msg = f"Schema in {piece} was different.\n" + str(e)
                        raise BodoError(msg)

                t0 = time.time()
                row_count = frag.scanner(
                    schema=dataset_.schema,
                    filter=expr_filters,
                    use_threads=True,
                ).count_rows()
                ds_scan_time += time.time() - t0
                piece._bodo_num_rows = row_count
                total_rows_chunk += row_count
                total_row_groups_chunk += frag.num_row_groups
                total_row_groups_size_chunk += sum(
                    rg.total_byte_size for rg in frag.row_groups
                )

        except Exception as e:
            error = e

        # synchronize error state
        if comm.allreduce(error is not None, op=MPI.LOR):
            for error in comm.allgather(error):
                if error:
                    if isinstance(fpath, list) and isinstance(
                        error, (OSError, FileNotFoundError)
                    ):
                        raise BodoError(str(error) + list_of_files_error_msg)
                    raise error

        if validate_schema:
            valid = comm.allreduce(valid, op=MPI.LAND)
            if not valid:  # pragma: no cover
                raise BodoError("Schema in parquet files don't match")

        if get_row_counts:
            dataset._bodo_total_rows = comm.allreduce(total_rows_chunk, op=MPI.SUM)
            total_num_row_groups = comm.allreduce(total_row_groups_chunk, op=MPI.SUM)
            total_row_groups_size = comm.allreduce(
                total_row_groups_size_chunk, op=MPI.SUM
            )
            pieces_rows = np.array([p._bodo_num_rows for p in dataset.pieces])
            # communicate row counts to everyone
            pieces_rows = comm.allreduce(pieces_rows, op=MPI.SUM)
            for p, nrows in zip(dataset.pieces, pieces_rows):
                p._bodo_num_rows = nrows
            if (
                is_parallel
                and bodo.get_rank() == 0
                and total_num_row_groups < bodo.get_size()
                and total_num_row_groups != 0
            ):
                warnings.warn(
                    BodoWarning(
                        f"""Total number of row groups in parquet dataset {fpath} ({total_num_row_groups}) is too small for effective IO parallelization.
For best performance the number of row groups should be greater than the number of workers ({bodo.get_size()}). For more details, refer to
https://docs.bodo.ai/latest/file_io/#parquet-section.
"""
                    )
                )
            # print a warning if average row group size < 1 MB and reading from remote filesystem
            if total_num_row_groups == 0:
                avg_row_group_size_bytes = 0
            else:
                avg_row_group_size_bytes = total_row_groups_size // total_num_row_groups
            if (
                bodo.get_rank() == 0
                and total_row_groups_size >= 20 * 1048576
                and avg_row_group_size_bytes < 1048576
                and protocol in REMOTE_FILESYSTEMS
            ):
                warnings.warn(
                    BodoWarning(
                        f"""Parquet average row group size is small ({avg_row_group_size_bytes} bytes) and can have negative impact on performance when reading from remote sources"""
                    )
                )
            if tracing.is_tracing():
                ev_row_counts.add_attribute(
                    "g_total_num_row_groups", total_num_row_groups
                )
                ev_row_counts.add_attribute("total_scan_time", ds_scan_time)
                # get 5-number summary for rowcounts:
                # (min, max, 25, 50 -median-, 75 percentiles)
                data = np.array([p._bodo_num_rows for p in dataset.pieces])
                quartiles = np.percentile(data, [25, 50, 75])
                ev_row_counts.add_attribute("g_row_counts_min", data.min())
                ev_row_counts.add_attribute("g_row_counts_Q1", quartiles[0])
                ev_row_counts.add_attribute("g_row_counts_median", quartiles[1])
                ev_row_counts.add_attribute("g_row_counts_Q3", quartiles[2])
                ev_row_counts.add_attribute("g_row_counts_max", data.max())
                ev_row_counts.add_attribute("g_row_counts_mean", data.mean())
                ev_row_counts.add_attribute("g_row_counts_std", data.std())
                ev_row_counts.add_attribute("g_row_counts_sum", data.sum())
                ev_row_counts.finalize()

    if read_categories:
        _add_categories_to_pq_dataset(dataset)

    if get_row_counts:
        ev.finalize()

    if validate_schema and is_parallel:
        if tracing.is_tracing():
            ev_unify_schemas = tracing.Event("unify_schemas_across_ranks")
        error = None

        try:
            dataset.schema = comm.allreduce(
                dataset.schema, bodo.io.helpers.pa_schema_unify_mpi_op
            )
        except Exception as e:
            error = e

        if tracing.is_tracing():
            ev_unify_schemas.finalize()

        # synchronize error state
        if comm.allreduce(error is not None, op=MPI.LOR):
            for error in comm.allgather(error):
                if error:
                    msg = f"Schema in some files were different.\n" + str(error)
                    raise BodoError(msg)

    return dataset


def get_scanner_batches(
    fpaths,
    expr_filters,
    selected_fields,
    avg_num_pieces,
    is_parallel,
    filesystem,
    str_as_dict_cols,
    start_offset,  # starting row offset in the pieces this process is going to read
    rows_to_read,  # total number of rows this process is going to read
    partitioning,
    schema,
):
    """return RecordBatchReader for dataset of 'fpaths' that contain the rows
    that match expr_filters (or all rows if expr_filters is None). Only project the
    fields in selected_fields"""
    import pyarrow as pa

    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count == 0:
        cpu_count = 2
    default_threads = min(int(os.environ.get("BODO_MIN_IO_THREADS", 4)), cpu_count)
    max_threads = min(int(os.environ.get("BODO_MAX_IO_THREADS", 16)), cpu_count)
    if (
        is_parallel
        and len(fpaths) > max_threads
        and len(fpaths) / avg_num_pieces >= 2.0
    ):
        # assign more threads to ranks that have to read
        # many more files than others
        pa.set_io_thread_count(max_threads)
        pa.set_cpu_count(max_threads)
    else:
        pa.set_io_thread_count(default_threads)
        pa.set_cpu_count(default_threads)
    pq_format = ds.ParquetFileFormat(dictionary_columns=str_as_dict_cols)
    # set columns to be read as dictionary encoded in schema
    dict_col_set = set(str_as_dict_cols)
    for i, name in enumerate(schema.names):
        if name in dict_col_set:
            old_field = schema.field(i)
            new_field = pa.field(
                name, pa.dictionary(pa.int32(), old_field.type), old_field.nullable
            )
            schema = schema.remove(i).insert(i, new_field)
    dataset = ds.dataset(
        fpaths,
        filesystem=filesystem,
        partitioning=partitioning,
        schema=schema,
        format=pq_format,
    )
    col_names = dataset.schema.names
    selected_names = [col_names[field_num] for field_num in selected_fields]

    # ------- row group filtering -------
    # Ranks typically will not read all the row groups from their list of
    # files (they will skip some rows at the beginning of the first file and
    # some rows at the end of the last one).
    # To make sure this rank only reads from the minimum necessary row groups,
    # we can create a new dataset object composed of row group fragments
    # instead of file fragments. We need to do it like this because Arrow's
    # scanner doesn't support skipping rows.
    # For this approach, we need to get row group metadata which can be very
    # expensive when reading from remote filesystems. Also, row group filtering
    # typically only benefits when the rank reads from a small set of files
    # (since the filtering only applies to the first and last file).
    # So we only filter based on this heuristic:
    # Filter row groups if the list of files is very small, or if it is <= 10
    # and this rank needs to skip rows of the first file
    filter_row_groups = len(fpaths) <= 3 or (start_offset > 0 and len(fpaths) <= 10)
    if filter_row_groups and (expr_filters is None):
        # TODO see if getting row counts with filter pushdown could be worthwhile
        # in some specific cases, and integrate that into the above heuristic
        new_frags = []
        # total number of rows of all the row groups we iterate through
        count_rows = 0
        # track total rows that this rank will read from row groups we iterate
        # through
        rows_added = 0
        for frag in dataset.get_fragments():  # each fragment is a parquet file
            # For reference, this is basically the same logic as in
            # ArrowDataframeReader::init_arrow_reader() and just adapted from there.
            # Get the file's row groups that this rank will read from
            row_group_ids = []
            for rg in frag.row_groups:
                num_rows_rg = rg.num_rows
                if start_offset < count_rows + num_rows_rg:
                    # rank needs to read from this row group
                    if rows_added == 0:
                        # this is the first row group the rank will read from
                        start_row_first_rg = start_offset - count_rows
                        rows_added_from_rg = min(
                            num_rows_rg - start_row_first_rg, rows_to_read
                        )
                    else:
                        rows_added_from_rg = min(num_rows_rg, rows_to_read - rows_added)
                    rows_added += rows_added_from_rg
                    row_group_ids.append(rg.id)
                count_rows += num_rows_rg
                if rows_added == rows_to_read:
                    break
            # XXX frag.subset(row_group_ids) is expensive on remote filesystems
            # with datasets composed of many files and row groups
            new_frags.append(frag.subset(row_group_ids=row_group_ids))
            if rows_added == rows_to_read:
                break
        dataset = ds.FileSystemDataset(
            new_frags, dataset.schema, pq_format, filesystem=dataset.filesystem
        )
        # The starting offset the Parquet reader knows about is from the first
        # file, not the first row group, so we need to communicate this back to C++
        start_offset = start_row_first_rg

    reader = dataset.scanner(
        columns=selected_names, filter=expr_filters, use_threads=True
    ).to_reader()
    return dataset, reader, start_offset


# XXX Move this to ParquetDataset class?
def _add_categories_to_pq_dataset(pq_dataset):
    """adds categorical values for each categorical column to the Parquet dataset
    as '_category_info' attribute
    """
    import pyarrow as pa
    from mpi4py import MPI

    # NOTE: shouldn't be possible
    if len(pq_dataset.pieces) < 1:  # pragma: no cover
        raise BodoError(
            "No pieces found in Parquet dataset. Cannot get read categorical values"
        )

    pa_schema = pq_dataset.schema
    cat_col_names = [
        c
        for c in pa_schema.names
        if isinstance(pa_schema.field(c).type, pa.DictionaryType)
        and c not in pq_dataset.partition_names
    ]

    # avoid more work if no categorical columns
    if len(cat_col_names) == 0:
        pq_dataset._category_info = {}
        return

    comm = MPI.COMM_WORLD
    if bodo.get_rank() == 0:
        try:
            # read categorical values from first row group of first file
            table_sample = pq_dataset.pieces[0].frag.head(100, columns=cat_col_names)
            # NOTE: assuming DictionaryArray has only one chunk
            category_info = {
                c: tuple(table_sample.column(c).chunk(0).dictionary.to_pylist())
                for c in cat_col_names
            }
            del table_sample  # release I/O resources ASAP
        except Exception as e:
            comm.bcast(e)
            raise e

        comm.bcast(category_info)
    else:
        category_info = comm.bcast(None)
        if isinstance(category_info, Exception):  # pragma: no cover
            error = category_info
            raise error

    pq_dataset._category_info = category_info


def get_pandas_metadata(schema, num_pieces):
    # find pandas index column if any
    # TODO: other pandas metadata like dtypes needed?
    # https://pandas.pydata.org/pandas-docs/stable/development/developer.html
    index_col = None
    # column_name -> is_nullable (or None if unknown)
    nullable_from_metadata = defaultdict(lambda: None)
    key = b"pandas"
    if schema.metadata is not None and key in schema.metadata:
        import json

        pandas_metadata = json.loads(schema.metadata[key].decode("utf8"))
        n_indices = len(pandas_metadata["index_columns"])
        if n_indices > 1:
            raise BodoError("read_parquet: MultiIndex not supported yet")
        index_col = pandas_metadata["index_columns"][0] if n_indices else None
        # ignore non-str/dict index metadata
        if not isinstance(index_col, str) and not isinstance(index_col, dict):
            index_col = None

        for col_dict in pandas_metadata["columns"]:
            col_name = col_dict["name"]
            if col_dict["pandas_type"].startswith("int") and col_name is not None:
                if col_dict["numpy_type"].startswith("Int"):
                    nullable_from_metadata[col_name] = True
                else:
                    nullable_from_metadata[col_name] = False
    return index_col, nullable_from_metadata


def get_str_columns_from_pa_schema(pa_schema):
    """
    Get the list of string type columns in the schema.
    """
    str_columns = []
    for col_name in pa_schema.names:
        field = pa_schema.field(col_name)
        if field.type in (pa.string(), pa.large_string()):
            str_columns.append(col_name)
    return str_columns


def determine_str_as_dict_columns(pq_dataset, pa_schema, str_columns):
    """
    Determine which string columns (str_columns) should be read by Arrow as
    dictionary encoded arrays, based on this heuristic:
      calculating the ratio of total_uncompressed_size of the column vs number
      of values.
      If the ratio is less than READ_STR_AS_DICT_THRESHOLD we read as DICT.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    if len(str_columns) == 0:
        return set()  # no string as dict columns

    # We don't want to open every file at compile time, so instead we will open
    # a sample of N files where N is the number of ranks. Each rank looks at
    # the metadata of a different random file
    if len(pq_dataset.pieces) > bodo.get_size():
        import random

        random.seed(37)
        pieces = random.sample(pq_dataset.pieces, bodo.get_size())
    else:
        pieces = pq_dataset.pieces
    total_uncompressed_sizes = np.zeros(len(str_columns), dtype=np.int64)
    total_uncompressed_sizes_recv = np.zeros(len(str_columns), dtype=np.int64)
    if bodo.get_rank() < len(pieces):
        piece = pieces[bodo.get_rank()]
        try:
            metadata = piece.metadata
            for i in range(piece.num_row_groups):
                for j, col_name in enumerate(str_columns):
                    idx = pa_schema.get_field_index(col_name)
                    total_uncompressed_sizes[j] += (
                        metadata.row_group(i).column(idx).total_uncompressed_size
                    )
            num_rows = metadata.num_rows
        except Exception as e:
            if isinstance(e, (OSError, FileNotFoundError)):
                # skip the path that produced the error (error will be reported at runtime)
                num_rows = 0
            else:
                raise
    else:
        num_rows = 0
    total_rows = comm.allreduce(num_rows, op=MPI.SUM)
    if total_rows == 0:
        return set()  # no string as dict columns
    comm.Allreduce(total_uncompressed_sizes, total_uncompressed_sizes_recv, op=MPI.SUM)
    str_column_metrics = total_uncompressed_sizes_recv / total_rows
    str_as_dict = set()
    for i, metric in enumerate(str_column_metrics):
        if metric < READ_STR_AS_DICT_THRESHOLD:
            col_name = str_columns[i][0]
            str_as_dict.add(col_name)
    return str_as_dict


def parquet_file_schema(
    file_name,
    selected_columns,
    storage_options=None,
    input_file_name_col=None,
    read_as_dict_cols=None,
):
    """get parquet schema from file using Parquet dataset and Arrow APIs"""
    col_names = []
    col_types = []

    # during compilation we only need the schema and it has to be the same for
    # all processes, so we can set parallel=True to just have rank 0 read
    # the dataset information and broadcast to others
    pq_dataset = get_parquet_dataset(
        file_name,
        get_row_counts=False,
        storage_options=storage_options,
        read_categories=True,
    )

    partition_names = pq_dataset.partition_names
    pa_schema = pq_dataset.schema
    num_pieces = len(pq_dataset.pieces)

    # Get list of string columns
    str_columns = get_str_columns_from_pa_schema(pa_schema)
    # Convert to set (easier for set operations like intersect and union)
    str_columns_set = set(str_columns)
    if read_as_dict_cols is None:
        read_as_dict_cols = []
    read_as_dict_cols = set(read_as_dict_cols)
    # If user-provided list has any columns that are not string
    # type, show a warning.
    non_str_columns_in_read_as_dict_cols = read_as_dict_cols - str_columns_set
    if len(non_str_columns_in_read_as_dict_cols) > 0:
        if bodo.get_rank() == 0:
            warnings.warn(
                f"The following columns are not of datatype string and hence cannot be read with dictionary encoding: {non_str_columns_in_read_as_dict_cols}",
                bodo.utils.typing.BodoWarning,
            )
    # Remove non-string columns from read_as_dict_cols
    read_as_dict_cols.intersection_update(str_columns_set)
    # Remove read_as_dict_cols from str_columns (no need to compute heuristic on these)
    str_columns_set = str_columns_set - read_as_dict_cols
    # Match the list with the set. We've only removed entries, so a filter is sufficient.
    # Order of columns in the list is important between different ranks,
    # so either we do this, or sort.
    str_columns = [x for x in str_columns if x in str_columns_set]
    # Get the set of columns to be read with dictionary encoding based on heuristic
    str_as_dict: set = determine_str_as_dict_columns(pq_dataset, pa_schema, str_columns)
    # Add user-provided columns to the list
    str_as_dict.update(read_as_dict_cols)

    # NOTE: use arrow schema instead of the dataset schema to avoid issues with
    # names of list types columns (arrow 0.17.0)
    # col_names is an array that contains all the column's name and
    # index's name if there is one, otherwise "__index__level_0"
    # If there is no index at all, col_names will not include anything.
    col_names = pa_schema.names
    index_col, nullable_from_metadata = get_pandas_metadata(pa_schema, num_pieces)
    col_types_total = []
    is_supported_list = []
    arrow_types = []
    for i, c in enumerate(col_names):
        if c in partition_names:
            continue
        field = pa_schema.field(c)
        dtype, supported = _get_numba_typ_from_pa_typ(
            field,
            c == index_col,
            nullable_from_metadata[c],
            pq_dataset._category_info,
            str_as_dict=c in str_as_dict,
        )
        col_types_total.append(dtype)
        is_supported_list.append(supported)
        # Store the unsupported arrow type for future
        # error messages.
        arrow_types.append(field.type)

    # add partition column data types if any
    if partition_names:
        col_types_total += [
            _get_partition_cat_dtype(pq_dataset.partitioning_dictionaries[i])
            for i in range(len(partition_names))
        ]
        # All partition column types are supported by default.
        is_supported_list.extend([True] * len(partition_names))
        # Extend arrow_types for consistency. Here we use None
        # because none of these are actually in the pq file.
        arrow_types.extend([None] * len(partition_names))

    # add input_file_name column data type if one is specified
    if input_file_name_col is not None:
        col_names += [input_file_name_col]
        col_types_total += [dict_str_arr_type]
        # input_file_name column is a dictionary-encoded string array which is supported by default.
        is_supported_list.append(True)
        # Extend arrow_types for consistency. Here we use None
        # because this column isn't actually in the pq file.
        arrow_types.append(None)

    # Map column names to index to allow efficient search
    col_names_map = {c: i for i, c in enumerate(col_names)}

    # if no selected columns, set it to all of them.
    if selected_columns is None:
        selected_columns = col_names

    # make sure selected columns are in the schema
    for c in selected_columns:
        if c not in col_names_map:
            raise BodoError(f"Selected column {c} not in Parquet file schema")
    if (
        index_col
        and not isinstance(index_col, dict)
        and index_col not in selected_columns
    ):
        # if index_col is "__index__level_0" or some other name, append it.
        # If the index column is not selected when reading parquet, the index
        # should still be included.
        selected_columns.append(index_col)

    col_names = selected_columns
    col_indices = []
    col_types = []
    unsupported_columns = []
    unsupported_arrow_types = []
    for i, c in enumerate(col_names):
        col_idx = col_names_map[c]
        col_indices.append(col_idx)
        col_types.append(col_types_total[col_idx])
        if not is_supported_list[col_idx]:
            unsupported_columns.append(i)
            unsupported_arrow_types.append(arrow_types[col_idx])

    # TODO: close file?
    return (
        col_names,
        col_types,
        index_col,
        col_indices,
        partition_names,
        unsupported_columns,
        unsupported_arrow_types,
    )


def _get_partition_cat_dtype(dictionary):
    """get categorical dtype for Parquet partition set"""
    # using 'dictionary' instead of 'keys' attribute since 'keys' may not have the
    # right data type (e.g. string instead of int64)
    assert dictionary is not None
    S = dictionary.to_pandas()
    elem_type = bodo.typeof(S).dtype
    if isinstance(elem_type, (types.Integer)):
        cat_dtype = PDCategoricalDtype(tuple(S), elem_type, False, int_type=elem_type)
    else:
        cat_dtype = PDCategoricalDtype(tuple(S), elem_type, False)
    return CategoricalArrayType(cat_dtype)


_pq_read = types.ExternalFunction(
    "pq_read",
    table_type(
        read_parquet_fpath_type,
        types.boolean,
        parquet_predicate_type,  # dnf filters
        parquet_predicate_type,  # expr filters
        storage_options_dict_type,
        types.int64,
        types.voidptr,
        types.int32,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.int32,
        types.voidptr,
        types.int32,
        types.voidptr,
        types.boolean,
    ),
)

from llvmlite import ir as lir
from numba.core import cgutils

if bodo.utils.utils.has_pyarrow():
    from bodo.io import arrow_cpp

    ll.add_symbol("pq_read", arrow_cpp.pq_read)
    ll.add_symbol("pq_write", arrow_cpp.pq_write)
    ll.add_symbol("pq_write_partitioned", arrow_cpp.pq_write_partitioned)


############################ parquet write table #############################


@intrinsic
def parquet_write_table_cpp(
    typingctx,
    filename_t,
    table_t,
    col_names_t,
    index_t,
    write_index,
    metadata_t,
    compression_t,
    is_parallel_t,
    write_range_index,
    start,
    stop,
    step,
    name,
    bucket_region,
    row_group_size,
    file_prefix,
):
    """
    Call C++ parquet write function
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(64),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(32),
                lir.IntType(32),
                lir.IntType(32),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(builder.module, fnty, name="pq_write")
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        types.int64(
            types.voidptr,
            table_t,
            col_names_t,
            index_t,
            types.boolean,
            types.voidptr,
            types.voidptr,
            types.boolean,
            types.boolean,
            types.int32,
            types.int32,
            types.int32,
            types.voidptr,
            types.voidptr,
            types.int64,
            types.voidptr,
        ),
        codegen,
    )


@intrinsic
def parquet_write_table_partitioned_cpp(
    typingctx,
    filename_t,
    data_table_t,
    col_names_t,
    col_names_no_partitions_t,
    cat_table_t,
    part_col_idxs_t,
    num_part_col_t,
    compression_t,
    is_parallel_t,
    bucket_region,
    row_group_size,
    file_prefix,
):
    """
    Call C++ parquet write partitioned function
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="pq_write_partitioned"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    return (
        types.void(
            types.voidptr,
            data_table_t,
            col_names_t,
            col_names_no_partitions_t,
            cat_table_t,
            types.voidptr,
            types.int32,
            types.voidptr,
            types.boolean,
            types.voidptr,
            types.int64,
            types.voidptr,
        ),
        codegen,
    )
