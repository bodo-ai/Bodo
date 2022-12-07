# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""IR node for the parquet data access"""

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
from numba.core import ir, ir_utils, typeinfer, types
from numba.core.ir_utils import (
    compile_to_numba_ir,
    get_definition,
    guard,
    mk_unique_var,
    next_label,
    replace_arg_nodes,
)
from numba.extending import NativeValue, models, register_model, unbox

import bodo
import bodo.ir.connector
from bodo.hiframes.table import Table, TableType  # noqa
from bodo.io.fs_io import (
    get_storage_options_pyobject,
    storage_options_dict_type,
)
from bodo.io.helpers import is_nullable
from bodo.io.parquet_pio import (
    ParquetFileInfo,
    get_filters_pyobject,
    parquet_file_schema,
    parquet_predicate_type,
)
from bodo.libs.array import (
    cpp_table_to_py_table,
    delete_table,
    info_from_table,
    info_to_array,
    table_type,
)
from bodo.libs.dict_arr_ext import dict_str_arr_type
from bodo.libs.str_ext import unicode_to_utf8
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.transforms.table_column_del_pass import (
    ir_extension_table_column_use,
    remove_dead_column_extensions,
)
from bodo.utils.transform import get_const_value
from bodo.utils.typing import BodoError, FilenameType
from bodo.utils.utils import (
    check_and_propagate_cpp_exception,
    numba_to_c_type,
    sanitize_varname,
)


class ReadParquetFilepathType(types.Opaque):
    """Type for file path object passed to C++. It is just a Python object passed
    as a pointer to C++ (can be Python list of strings or Python string)
    """

    def __init__(self):
        super(ReadParquetFilepathType, self).__init__(name="ReadParquetFilepathType")


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
        use_hive=True,
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
                ) = parquet_file_schema(
                    file_name_str,
                    columns,
                    storage_options,
                    input_file_name_col,
                    read_as_dict_cols,
                    use_hive,
                )
        else:
            col_names_total = list(table_types.keys())
            # Create a map for efficient index lookup
            col_names_total_map = {c: i for i, c in enumerate(col_names_total)}
            col_types_total = [t for t in table_types.values()]
            index_col = "index" if "index" in col_names_total_map else None
            # TODO: allow specifying types of only selected columns
            selected_columns = col_names_total if columns is None else columns
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
                index_column_index,
                index_column_type,
                input_file_name_col,
                unsupported_columns,
                unsupported_arrow_types,
                use_hive,
            )
        ]

        return col_names, data_arrs, index_col, nodes, col_types, index_column_type


class ParquetReader(ir.Stmt):
    def __init__(
        self,
        file_name,
        df_out,
        col_names,
        col_indices,
        out_types,
        out_vars,
        loc,
        partition_names,
        # These are the same storage_options that would be passed to pandas
        storage_options,
        index_column_index,
        index_column_type,
        input_file_name_col,
        unsupported_columns,
        unsupported_arrow_types,
        use_hive,
    ):
        self.connector_typ = "parquet"
        self.file_name = file_name
        self.df_out = df_out  # used only for printing
        self.df_colnames = col_names
        self.col_indices = col_indices
        self.out_types = out_types
        # Original out types + columns are maintained even if columns are pruned.
        # This is maintained in case we need type info for filter pushdown and
        # the column has been eliminated.
        # For example, if our Pandas code was:
        # def ex(filename):
        #     df = pd.read_parquet(filename)
        #     df = df[df.A > 1]
        #     return df[["B", "C"]]
        # Then DCE should remove all columns from df_colnames/out_types except B and C,
        # but we still need to the type of column A to determine if we need to generate
        # a cast inside the arrow filters.
        self.original_out_types = out_types
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
        # Is the variable currently alive. This should be replaced with more
        # robust handling in connectors.
        self.is_live_table = True
        self.use_hive = use_hive

    def __repr__(self):  # pragma: no cover
        # TODO
        return "({}) = ReadParquet({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
            self.df_out,
            self.file_name.name,
            self.df_colnames,
            self.col_indices,
            self.out_types,
            self.original_out_types,
            self.original_df_colnames,
            self.out_vars,
            self.partition_names,
            self.filters,
            self.storage_options,
            self.index_column_index,
            self.index_column_type,
            self.out_used_cols,
            self.input_file_name_col,
            self.unsupported_columns,
            self.unsupported_arrow_types,
        )


def remove_dead_pq(
    pq_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):
    """
    Function that eliminates parquet reader variables when they
    are no longer live.
    """
    table_var = pq_node.out_vars[0].name
    index_var = pq_node.out_vars[1].name
    if table_var not in lives and index_var not in lives:
        # If neither the table or index is live, remove the node.
        return None
    elif table_var not in lives:
        # If table isn't live we only want to load the index.
        # To do this we should mark the col_indices as empty
        pq_node.col_indices = []
        pq_node.df_colnames = []
        pq_node.out_used_cols = []
        pq_node.is_live_table = False
    elif index_var not in lives:
        # If the index_var not in lives we don't load the index.
        # To do this we mark the index_column_index as None
        pq_node.index_column_index = None
        pq_node.index_column_type = types.none
    # TODO: Update the usecols if only 1 of the variables is live.
    return pq_node


def pq_remove_dead_column(pq_node, column_live_map, equiv_vars, typemap):
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
        pq_node.use_hive,
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
    use_hive: bool,
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
    func_text += (
        f"    total_rows_np = np.array([0], dtype=np.int64)\n"
        f"    out_table = pq_read(\n"
        f"        fname_py,\n"
        f"        {is_parallel},\n"
        f"        dnf_filters,\n"
        f"        expr_filters,\n"
        f"        storage_options_py,\n"
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
        func_text += f"        0, 0, 0,\n"
    if len(str_as_dict_cols) > 0:
        # TODO pass array as global to function instead?
        func_text += f"        np.array({str_as_dict_cols}, dtype=np.int32).ctypes, {len(str_as_dict_cols)},\n"
    else:
        func_text += f"        0, 0,\n"
    func_text += f"        total_rows_np.ctypes,\n"
    # The C++ code only needs a flag
    func_text += f"        {input_file_name_col is not None},\n"
    func_text += f"        {use_hive},\n"
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
        if len(out_used_cols) == 0:
            # Set the table length using the total rows if don't load any columns
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


@numba.njit
def get_fname_pyobject(fname):
    """Convert fname native object (which can be a string or a list of strings)
    to its corresponding PyObject by going through unboxing and boxing"""
    with numba.objmode(fname_py="read_parquet_fpath_type"):
        fname_py = fname
    return fname_py


numba.parfors.array_analysis.array_analysis_extensions[
    ParquetReader
] = bodo.ir.connector.connector_array_analysis
distributed_analysis.distributed_analysis_extensions[
    ParquetReader
] = bodo.ir.connector.connector_distributed_analysis
typeinfer.typeinfer_extensions[ParquetReader] = bodo.ir.connector.connector_typeinfer
ir_utils.visit_vars_extensions[ParquetReader] = bodo.ir.connector.visit_vars_connector
ir_utils.remove_dead_extensions[ParquetReader] = remove_dead_pq
numba.core.analysis.ir_extension_usedefs[
    ParquetReader
] = bodo.ir.connector.connector_usedefs
ir_utils.copy_propagate_extensions[
    ParquetReader
] = bodo.ir.connector.get_copies_connector
ir_utils.apply_copy_propagate_extensions[
    ParquetReader
] = bodo.ir.connector.apply_copies_connector
ir_utils.build_defs_extensions[
    ParquetReader
] = bodo.ir.connector.build_connector_definitions
remove_dead_column_extensions[ParquetReader] = pq_remove_dead_column
ir_extension_table_column_use[
    ParquetReader
] = bodo.ir.connector.connector_table_column_use
distributed_pass.distributed_run_extensions[ParquetReader] = pq_distributed_run


if bodo.utils.utils.has_pyarrow():
    from bodo.io import arrow_cpp

    ll.add_symbol("pq_read", arrow_cpp.pq_read)

_pq_read = types.ExternalFunction(
    "pq_read",
    table_type(
        read_parquet_fpath_type,  # path
        types.boolean,  # parallel
        parquet_predicate_type,  # dnf_filters
        parquet_predicate_type,  # expr_filters
        storage_options_dict_type,  # storage_options
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
