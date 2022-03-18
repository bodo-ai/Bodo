# Copyright (C) 2019 Bodo Inc. All rights reserved.
import os
import warnings
from collections import defaultdict

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
import pyarrow  # noqa
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
    intrinsic,
    models,
    overload,
    register_model,
    unbox,
)
from pyarrow import null

import bodo
import bodo.ir.parquet_ext
import bodo.utils.tracing as tracing
from bodo.hiframes.datetime_date_ext import (
    datetime_date_array_type,
    datetime_date_type,
)
from bodo.hiframes.pd_categorical_ext import (
    CategoricalArrayType,
    PDCategoricalDtype,
)
from bodo.hiframes.table import TableType
from bodo.io.fs_io import get_hdfs_fs, get_s3_fs_from_path, get_s3_subtree_fs
from bodo.libs.array import (
    cpp_table_to_py_table,
    delete_table,
    info_from_table,
    info_to_array,
    table_type,
)
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.binary_arr_ext import binary_array_type, bytes_type
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.dict_arr_ext import dict_str_arr_type
from bodo.libs.distributed_api import get_end, get_start
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import string_type, unicode_to_utf8
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.transforms import distributed_pass
from bodo.utils.transform import get_const_value
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    FileInfo,
    get_overload_const_str,
    get_overload_constant_dict,
)
from bodo.utils.utils import (
    check_and_propagate_cpp_exception,
    numba_to_c_type,
    sanitize_varname,
)

# read Arrow Int columns as nullable int array (IntegerArrayType)
use_nullable_int_arr = True


from urllib.parse import urlparse

import bodo.io.pa_parquet

REMOTE_FILESYSTEMS = {"s3", "gcs", "gs", "http", "hdfs", "abfs", "abfss"}
# the ratio of total_uncompressed_size of a Parquet string column vs number of values,
# below which we read as dictionary-encoded string array
READ_STR_AS_DICT_THRESHOLD = 1.0


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


class StorageOptionsDictType(types.Opaque):
    def __init__(self):
        super(StorageOptionsDictType, self).__init__(name="StorageOptionsDictType")


storage_options_dict_type = StorageOptionsDictType()
types.storage_options_dict_type = storage_options_dict_type
register_model(StorageOptionsDictType)(models.OpaqueModel)


@unbox(StorageOptionsDictType)
def unbox_storage_options_dict_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return NativeValue(val)


class ParquetFileInfo(FileInfo):
    """FileInfo object passed to ForceLiteralArg for
    file name arguments that refer to a parquet dataset"""

    def __init__(self, columns, storage_options=None, input_file_name_col=None):
        self.columns = columns  # columns to select from parquet dataset
        self.storage_options = storage_options
        self.input_file_name_col = input_file_name_col
        super().__init__()

    def _get_schema(self, fname):
        try:
            return parquet_file_schema(
                fname,
                selected_columns=self.columns,
                storage_options=self.storage_options,
                input_file_name_col=self.input_file_name_col,
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
        self, file_name, lhs, columns, storage_options=None, input_file_name_col=None
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
            msg = "Parquet schema not available. Either path argument should be constant for Bodo to look at the file at compile time or schema should be provided. For more information, see: https://docs.bodo.ai/latest/source/programming_with_bodo/file_io.html#non-constant-filepaths"
            file_name_str = get_const_value(
                file_name,
                self.func_ir,
                msg,
                arg_types=self.args,
                file_info=ParquetFileInfo(
                    columns,
                    storage_options=storage_options,
                    input_file_name_col=input_file_name_col,
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


def determine_filter_cast(pq_node, typemap, filter_val, orig_colname_map):
    """
    Function that generates text for casts that need to be included
    in the filter when not automatically handled by Arrow. For example
    timestamp and string. This function returns two strings. In most cases
    one of the strings will be empty and the other contain the argument that
    should be cast. However, if we have a partition column, we always cast the
    partition column either to its original type or the new type.

    This is because of an issue in arrow where if a partion has a mix of integer
    and string names it won't look at the global types when Bodo processes per
    file (see test_read_partitions_string_int for an example).

    We opt to cast in the direction that keep maximum information, for
    example date -> timestamp rather than timestamp -> date.

    Right now we assume filter_val[0] is a column name and filter_val[2]
    is a scalar Var that is found in the typemap.
    """
    colname = filter_val[0]
    lhs_arr_typ = pq_node.original_out_types[orig_colname_map[colname]]
    lhs_scalar_typ = bodo.utils.typing.element_type(lhs_arr_typ)
    if colname in pq_node.partition_names:
        # Always cast partitions to protect again multiple types
        # see test_read_partitions_string_int.

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
    # If we do isin, then rhs_typ will be a list or set
    if isinstance(rhs_typ, (types.List, types.Set)):
        rhs_scalar_typ = rhs_typ.dtype
    else:
        rhs_scalar_typ = rhs_typ

    # Here we assume is_common_scalar_dtype conversions are common
    # enough that Arrow will support them, since these are conversions
    # like int -> float. TODO: Test
    if not bodo.utils.typing.is_common_scalar_dtype([lhs_scalar_typ, rhs_scalar_typ]):
        # If a cast is not implicit it must be in our white list.
        if not bodo.utils.typing.is_safe_arrow_cast(lhs_scalar_typ, rhs_scalar_typ):
            raise BodoError(
                f"Unsupported Arrow cast from {lhs_scalar_typ} to {rhs_scalar_typ} in filter pushdown. Please try a comparison that avoids casting the column."
            )
        # We always cast string -> other types
        # Only supported types should be string and timestamp
        if lhs_scalar_typ == types.unicode_type:
            return ".cast(pyarrow.timestamp('ns'), safe=False)", ""
        elif lhs_scalar_typ in (bodo.datetime64ns, bodo.pd_timestamp_type):
            if isinstance(rhs_typ, (types.List, types.Set)):  # pragma: no cover
                # This path should never be reached because we checked that
                # list/set doesn't contain Timestamp or datetime64 in typing pass.
                type_name = "list" if isinstance(rhs_typ, types.List) else "tuple"
                raise BodoError(
                    f"Cannot cast {type_name} values with isin filter pushdown."
                )
            return col_cast, ".cast(pyarrow.timestamp('ns'), safe=False)"

    return col_cast, ""


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
    extra_args = ""
    dnf_filter_str = "None"
    expr_filter_str = "None"

    filter_map, filter_vars = bodo.ir.connector.generate_filter_map(pq_node.filters)
    # If no filters use variables (i.e. all isna, then we still need to take this path)
    if pq_node.filters:
        # Create two formats for parquet. One using the old DNF format
        # which contains just the partition columns (partition pushdown)
        # and one using the more verbose expression format and all expressions
        # (predicate pushdown).
        # https://arrow.apache.org/docs/python/dataset.html#filtering-data
        #
        # partition pushdown requires the expressions to just contain Hive partitions
        # If any expressions are not hive partitions then the DNF filters is an AND of
        # filters expressions that are seen in ALL OR conditions
        #
        # For example if A, C are paritions and B is not
        #
        # Then (A & B) | (A & C) -> A (for the partition expressions)
        #
        dnf_or_conds = []
        expr_or_conds = []
        dnf_modified = False
        # If one of the filters isn't a partition column, then
        # we can only select the partition columns that are shared
        # across all possible or statements. Initialize this set to
        # None so we don't intersect with an empty set in the first OR.
        shared_expr_set = None
        # Create a mapping for faster column indexing
        orig_colname_map = {c: i for i, c in enumerate(pq_node.original_df_colnames)}
        for predicate in pq_node.filters:
            dnf_and_conds = []
            expr_and_conds = []
            and_expr_set = set()
            for v in predicate:
                # First update expr since these include all expressions.
                # If v[2] is a var we pass the variable at runtime,
                # otherwise we pass a constant (i.e. NULL) which
                # requires special handling
                if isinstance(v[2], ir.Var):
                    # expr conds don't do some of the casts that DNF expressions do (for example String and Timestamp).
                    # For known cases where we need to cast we generate the cast. For simpler code we return two strings,
                    # column_cast and scalar_cast.
                    column_cast, scalar_cast = determine_filter_cast(
                        pq_node, typemap, v, orig_colname_map
                    )
                    if v[1] == "in":
                        # Expected output for this format should like
                        # (ds.field('A').isin(py_var))
                        expr_and_conds.append(
                            f"(ds.field('{v[0]}').isin({filter_map[v[2].name]}))"
                        )
                    else:
                        # Expected output for this format should like
                        # (ds.field('A') > ds.scalar(py_var))
                        expr_and_conds.append(
                            f"(ds.field('{v[0]}'){column_cast} {v[1]} ds.scalar({filter_map[v[2].name]}){scalar_cast})"
                        )
                else:
                    # Currently the only constant expressions we support are IS [NOT] NULL
                    assert v[2] == "NULL", "unsupport constant used in filter pushdown"
                    if v[1] == "is not":
                        prefix = "~"
                    else:
                        prefix = ""
                    # Expected output for this format should like
                    # (~ds.field('A').is_null())
                    expr_and_conds.append(f"({prefix}ds.field('{v[0]}').is_null())")
                # Now handle the dnf section. We can only append a value if its not a constant
                # expression and is a partition column
                if v[0] in pq_node.partition_names and isinstance(v[2], ir.Var):
                    dnf_str = f"('{v[0]}', '{v[1]}', {filter_map[v[2].name]})"
                    dnf_and_conds.append(dnf_str)
                    and_expr_set.add(dnf_str)
                else:
                    # Mark this expression as modified.
                    dnf_modified = True

            if shared_expr_set is None:
                shared_expr_set = and_expr_set
            else:
                shared_expr_set.intersection_update(and_expr_set)

            # Group and according to the expected format
            dnf_and_str = ", ".join(dnf_and_conds)
            expr_and_str = " & ".join(expr_and_conds)
            # If all the filters are truncated we may have an empty string.
            if dnf_and_str:
                dnf_or_conds.append(f"[{dnf_and_str}]")
            expr_or_conds.append(f"({expr_and_str})")

        dnf_or_str = ", ".join(dnf_or_conds)
        expr_or_str = " | ".join(expr_or_conds)
        if dnf_modified:
            # If DNF filters are modified we produce an AND of the
            # common partition expressions.
            if shared_expr_set:
                # Sort the set to get consistent code on all ranks
                filters = sorted(shared_expr_set)
                dnf_filter_str = f"[[{', '.join(filters)}]]"
        elif dnf_or_str:
            # If the expression is not modified (and exist) we use the whole expression
            dnf_filter_str = f"[{dnf_or_str}]"
        expr_filter_str = f"({expr_or_str})"

        extra_args = ", ".join(filter_map.values())

    # get column variables
    arg_names = ", ".join(f"out{i}" for i in range(n_cols))
    func_text = f"def pq_impl(fname, {extra_args}):\n"
    # total_rows is used for setting total size variable below
    func_text += (
        f"    (total_rows, {arg_names},) = _pq_reader_py(fname, {extra_args})\n"
    )

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    pq_impl = loc_vars["pq_impl"]

    # Add debug info about column pruning
    if bodo.user_logging.get_verbose_level() >= 1:
        msg = "Finish column pruning on read_parquet node:\n%s\nColumns loaded %s\n"
        pq_source = pq_node.loc.strformat()
        pq_cols = [pq_node.df_colnames[i] for i in pq_node.type_usecol_offset]
        bodo.user_logging.log_message(
            "Column Pruning",
            msg,
            pq_source,
            pq_cols,
        )

    # parallel read flag
    parallel = False
    if array_dists is not None:
        # table is parallel
        table_varname = pq_node.out_vars[0].name
        parallel = array_dists[table_varname] in (
            distributed_pass.Distribution.OneD,
            distributed_pass.Distribution.OneD_Var,
        )
        index_varname = pq_node.out_vars[1].name
        # index array parallelism should match the table
        assert (
            typemap[index_varname] == types.none
            or not parallel
            or array_dists[index_varname]
            in (
                distributed_pass.Distribution.OneD,
                distributed_pass.Distribution.OneD_Var,
            )
        ), "pq data/index parallelization does not match"

    # Check for any unsupported columns still remaining
    if pq_node.unsupported_columns:
        used_cols_set = set(pq_node.type_usecol_offset)
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
        pq_node.type_usecol_offset,
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
        nodes[-1 - n_cols].target = meta_head_only_info[1]

    # assign output table
    nodes[-2].target = pq_node.out_vars[0]
    # assign output index array
    nodes[-1].target = pq_node.out_vars[1]

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


def get_storage_options_pyobject(storage_options):  # pragma: no cover
    pass


@overload(get_storage_options_pyobject, no_unliteral=True)
def overload_get_storage_options_pyobject(storage_options):
    """generate a pyobject for the storage_options to pass to C++"""
    storage_options_val = get_overload_constant_dict(storage_options)
    func_text = "def impl(storage_options):\n"
    func_text += (
        "  with numba.objmode(storage_options_py='storage_options_dict_type'):\n"
    )
    func_text += f"    storage_options_py = {str(storage_options_val)}\n"
    func_text += "  return storage_options_py\n"
    loc_vars = {}
    exec(func_text, globals(), loc_vars)
    return loc_vars["impl"]


def _gen_pq_reader_py(
    col_names,
    col_indices,
    type_usecol_offset,
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
):

    # a unique int used to create global variables with unique names
    call_id = next_label()

    comma = "," if extra_args else ""
    func_text = f"def pq_reader_py(fname,{extra_args}):\n"
    # if it's an s3 url, get the region and pass it into the c++ code
    func_text += f"    ev = bodo.utils.tracing.Event('read_parquet', {is_parallel})\n"
    func_text += f"    ev.add_attribute('fname', fname)\n"
    func_text += f"    bucket_region = bodo.io.fs_io.get_s3_bucket_region_njit(fname, parallel={is_parallel})\n"
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

    # If we aren't loading any column the table is dead
    is_dead_table = not type_usecol_offset

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
        and (col_names.index(input_file_name_col) in type_usecol_offset)
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
    # we only load the indices in type_usecol_offset.
    selected_cols = []
    partition_indices = set()
    cols_to_skip = partition_names + [input_file_name_col]
    for i in type_usecol_offset:
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
    # arrays require null bitmap and which don't

    def is_nullable(typ):
        return bodo.utils.utils.is_array_typ(typ, False) and (
            not isinstance(typ, types.Array)  # or is_dtype_nullable(typ.dtype)
        )

    # we need to load the nullable check in the same order as select columns. To do
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
    func_text += f"        unicode_to_utf8(bucket_region),\n"
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

    # handle index column
    index_arr = "None"
    index_arr_type = index_column_type
    py_table_type = TableType(tuple(out_types))
    if is_dead_table:
        py_table_type = types.none

    if index_column_index is not None:
        index_arr_ind = selected_cols_map[index_column_index]
        index_arr = f"info_to_array(info_from_table(out_table, {index_arr_ind}), index_arr_type)"
    func_text += f"    index_arr = {index_arr}\n"

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
            if j < len(type_usecol_offset) and i == type_usecol_offset[j]:
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

    if is_dead_table:
        func_text += "    T = None\n"
    else:
        func_text += f"    T = cpp_table_to_py_table(out_table, table_idx_{call_id}, py_table_type_{call_id})\n"
    func_text += f"    delete_table(out_table)\n"
    func_text += f"    total_rows = total_rows_np[0]\n"
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
    }

    exec(func_text, glbs, loc_vars)
    pq_reader_py = loc_vars["pq_reader_py"]

    jit_func = numba.njit(pq_reader_py, no_cpython_wrapper=True)
    return jit_func


import pyarrow as pa

_pa_numba_typ_map = {
    # boolean
    pa.bool_(): types.bool_,
    # signed int types
    pa.int8(): types.int8,
    pa.int16(): types.int16,
    pa.int32(): types.int32,
    pa.int64(): types.int64,
    # unsigned int types
    pa.uint8(): types.uint8,
    pa.uint16(): types.uint16,
    pa.uint32(): types.uint32,
    pa.uint64(): types.uint64,
    # float types (TODO: float16?)
    pa.float32(): types.float32,
    pa.float64(): types.float64,
    # String
    pa.string(): string_type,
    pa.binary(): bytes_type,
    # date
    pa.date32(): datetime_date_type,
    pa.date64(): types.NPDatetime("ns"),
    # time (TODO: time32, time64, ...)
    pa.timestamp("ns"): types.NPDatetime("ns"),
    pa.timestamp("us"): types.NPDatetime("ns"),
    pa.timestamp("ms"): types.NPDatetime("ns"),
    pa.timestamp("s"): types.NPDatetime("ns"),
    # all null column
    null(): string_type,  # map it to string_type, handle differently at runtime
}


def _get_numba_typ_from_pa_typ(
    pa_typ,
    is_index,
    nullable_from_metadata,
    category_info,
    str_as_dict=False,
):
    """return Bodo array type from pyarrow Field (column type) and if the type is supported.
    If a type is not support but can be adequately typed, we return that it isn't supported
    and later in compilation we will check if dead code/column elimination has successfully
    removed the column."""

    if isinstance(pa_typ.type, pa.ListType):
        # nullable_from_metadata is only used for non-nested Int arrays
        arr_typ, supported = _get_numba_typ_from_pa_typ(
            pa_typ.type.value_field, is_index, nullable_from_metadata, category_info
        )
        return ArrayItemArrayType(arr_typ), supported

    if isinstance(pa_typ.type, pa.StructType):
        child_types = []
        field_names = []
        supported = True
        for field in pa_typ.flatten():
            field_names.append(field.name.split(".")[-1])
            child_arr, child_supported = _get_numba_typ_from_pa_typ(
                field, is_index, nullable_from_metadata, category_info
            )
            child_types.append(child_arr)
            supported = supported and child_supported
        return StructArrayType(tuple(child_types), tuple(field_names)), supported

    # Decimal128Array type
    if isinstance(pa_typ.type, pa.Decimal128Type):
        return DecimalArrayType(pa_typ.type.precision, pa_typ.type.scale), True

    if str_as_dict:
        if pa_typ.type != pa.string():
            raise BodoError(f"Read as dictionary used for non-string column {pa_typ}")
        return dict_str_arr_type, True

    # Categorical data type
    if isinstance(pa_typ.type, pa.DictionaryType):
        # NOTE: non-string categories seems not possible as of Arrow 4.0
        if pa_typ.type.value_type != pa.string():  # pragma: no cover
            raise BodoError(
                f"Parquet Categorical data type should be string, not {pa_typ.type.value_type}"
            )
        # data type for storing codes
        int_type = _pa_numba_typ_map[pa_typ.type.index_type]
        cat_dtype = PDCategoricalDtype(
            category_info[pa_typ.name],
            bodo.string_type,
            pa_typ.type.ordered,
            int_type=int_type,
        )
        return CategoricalArrayType(cat_dtype), True

    if pa_typ.type not in _pa_numba_typ_map:
        # Timestamps with Timezones aren't supported inside Bodo. However,
        # they can be safely typed with regular Timestamps. Later in distributed_pass
        # for SQLReader and ParquetReader we check if these column still exist and
        # only then raise an exception.
        if isinstance(pa_typ.type, pa.lib.TimestampType) and pa_typ.type.tz is not None:
            dtype = types.NPDatetime("ns")
            supported = False
        else:
            raise BodoError("Arrow data type {} not supported yet".format(pa_typ.type))
    else:
        dtype = _pa_numba_typ_map[pa_typ.type]
        supported = True

    if dtype == datetime_date_type:
        return datetime_date_array_type, supported

    if dtype == bytes_type:
        return binary_array_type, supported

    arr_typ = string_array_type if dtype == string_type else types.Array(dtype, 1, "C")

    if dtype == types.bool_:
        arr_typ = boolean_array

    if nullable_from_metadata is not None:
        # do what metadata says
        _use_nullable_int_arr = nullable_from_metadata
    else:
        # use our global default
        _use_nullable_int_arr = use_nullable_int_arr
    # TODO: support nullable int for indices
    if (
        _use_nullable_int_arr
        and not is_index
        and isinstance(dtype, types.Integer)
        and pa_typ.nullable
    ):
        arr_typ = IntegerArrayType(dtype)

    return arr_typ, supported


def get_parquet_dataset(
    fpath,
    get_row_counts=True,
    dnf_filters=None,
    expr_filters=None,
    storage_options=None,
    read_categories=False,
    is_parallel=False,  # only used with get_row_counts=True
    tot_rows_to_read=None,
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
            fs.append(google_fs)
        elif protocol == "http":
            fs.append(fsspec.filesystem("http"))
        elif protocol in {"hdfs", "abfs", "abfss"}:  # pragma: no cover
            fs.append(
                get_hdfs_fs(fpath)
                if not isinstance(fpath, list)
                else get_hdfs_fs(fpath[0])
            )
        else:
            fs.append(None)
        return fs[0]

    def get_legacy_fs():
        """return filesystem with legacy interface that ParquetDataset with use_legacy_dataset=True can understand"""
        if protocol in {"s3", "hdfs", "abfs", "abfss"}:
            from fsspec.implementations.arrow import ArrowFSWrapper

            # fsspec wrapper makes it legacy API compatible
            return ArrowFSWrapper(getfs())
        else:
            return getfs()

    def glob(protocol, fs, path):
        """Return a possibly-empty list of path names that match glob pattern
        given by path"""
        if not protocol and fs is None:
            from fsspec.implementations.local import LocalFileSystem

            fs = LocalFileSystem()
        if isinstance(fs, pyarrow.fs.FileSystem):
            from fsspec.implementations.arrow import ArrowFSWrapper

            fs = ArrowFSWrapper(fs)
        try:
            if protocol in {"hdfs", "abfs", "abfss"}:  # pragma: no cover
                # HDFS filesystem is initialized with host:port info. Once
                # initialized, the filesystem needs the <protocol>://<host><port>
                # prefix removed to query and access files
                prefix = f"{protocol}://{parsed_url.netloc}"
                path = path[len(prefix) :]
            files = fs.glob(path)
            if protocol == "s3":
                # we need the s3:// prefix for later code to work correctly
                files = ["s3://" + f for f in files if not f.startswith("s3://")]
            elif protocol in {"hdfs", "abfs", "abfss"}:  # pragma: no cover
                # We add the prefix again because later code still expects it
                # TODO: we can maybe rewrite the code to avoid this
                files = [prefix + f for f in files]
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
        # There are issues with broadcasting the s3 filesystem object
        # as part of the ParquetDataset, so instead we initialize
        # the filesystem before the broadcast. The issues seem related
        # to pickling the filesystem. For example, for some reason
        # unpickling seems like it can cause incorrect credential handling
        # state in Arrow or AWS client.
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
                    ev_pq_ds.add_attribute("dnf_filter", str(dnf_filters))
            pa_default_io_thread_count = pa.io_thread_count()
            pa.set_io_thread_count(nthreads)

            if "*" in fpath:
                fpath = glob(protocol, getfs(), fpath)
            if protocol == "s3":
                # If there are issues accessing the s3 path (like wrong credentials)
                # fsspec will suppress the error and ParquetDataset() will ultimately
                # raise an exception claiming invalid path, which is not accurate.
                # To get the actual error, we directly query path info before calling
                # ParquetDataset().
                # NOTE: the result of fs.info(path) is cached the first time,
                # so subsequent calls (done inside ParquetDataset()) will not
                # have additional overhead
                if isinstance(fpath, list):
                    get_legacy_fs().info(fpath[0])
                else:
                    get_legacy_fs().info(fpath)
            if protocol in {"hdfs", "abfs", "abfss"}:  # pragma: no cover
                # HDFS filesystem is initialized with host:port info. Once
                # initialized, the filesystem needs the <protocol>://<host><port>
                # prefix removed to query and access files
                prefix = f"{protocol}://{parsed_url.netloc}"
                if isinstance(fpath, list):
                    fpath_arg = [f[len(prefix) :] for f in fpath]
                else:
                    fpath_arg = fpath[len(prefix) :]
            else:
                fpath_arg = fpath
            dataset = pq.ParquetDataset(
                fpath_arg,
                filesystem=get_legacy_fs(),  # need fs that works with use_legacy_dataset=True
                filters=None,
                use_legacy_dataset=True,  # To ensure that ParquetDataset and not ParquetDatasetV2 is used
                validate_schema=False,  # we do validation below if needed
                metadata_nthreads=nthreads,
            )
            # restore pyarrow default IO thread count
            pa.set_io_thread_count(pa_default_io_thread_count)
            dataset_schema = bodo.io.pa_parquet.get_dataset_schema(dataset)
            if dnf_filters:
                # Apply filters after getting the schema, because they might
                # remove all the pieces from the dataset which would make
                # get_dataset_schema() fail.
                # Use DNF, ParquetDataset does not support Expression filters
                if get_row_counts:
                    ev_pq_ds.add_attribute(
                        "num_pieces_before_filter", len(dataset.pieces)
                    )
                t0 = time.time()
                dataset._filter(dnf_filters)
                if get_row_counts:
                    ev_pq_ds.add_attribute("dnf_filter_time", time.time() - t0)
                    ev_pq_ds.add_attribute(
                        "num_pieces_after_filter", len(dataset.pieces)
                    )
            if get_row_counts:
                ev_pq_ds.finalize()

            # We don't want to send the filesystem because of the issues
            # mentioned above, so we set it to None. Note that this doesn't
            # seem to be enough to prevent sending it
            dataset._metadata.fs = None
        except Exception as e:
            comm.bcast(e)
            # See note in s3_list_dir_fnames
            # In some cases, OSError/FileNotFoundError can propagate back to numba and comes back as InternalError.
            # where numba errors are hidden from the user.
            # See [BE-1188] for an example
            # Raising a BodoError lets message comes back and seen by the user.
            raise BodoError(f"error from pyarrow: {type(e).__name__}: {str(e)}\n")

        if get_row_counts:
            ev_bcast = tracing.Event("bcast dataset")
        comm.bcast(dataset)
        comm.bcast(dataset_schema)
    else:
        if get_row_counts:
            ev_bcast = tracing.Event("bcast dataset")
        dataset = comm.bcast(None)
        if isinstance(dataset, Exception):  # pragma: no cover
            error = dataset
            raise BodoError(
                f"error from pyarrow: {type(error).__name__}: {str(error)}\n"
            )
        dataset_schema = comm.bcast(None)

    # Reinitialize the file system. The filesystem is needed at compile time
    # to get metadata from a sample of pieces in determine_str_as_dict_columns
    # and at runtime is needed to get metadata for rowcounts and row counts
    # for filter pushdown.
    if get_row_counts:
        filesystem = getfs()
    else:
        # Compile time needs the legacy filesystem.
        filesystem = get_legacy_fs()
    dataset._metadata.fs = filesystem

    if get_row_counts:
        ev_bcast.finalize()

    dataset._bodo_total_rows = 0
    if get_row_counts and tot_rows_to_read == 0:
        get_row_counts = validate_schema = False
        for p in dataset.pieces:
            p._bodo_num_rows = 0
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
        for p in pieces:
            p._bodo_num_rows = 0

        fpaths = [p.path for p in pieces[start:end]]
        if protocol == "s3":
            # We can call the Arrow dataset API passing a list of s3 file paths in two ways:
            # - Passing a s3 URL, removing the base path s3://my-bucket from the file paths, and calling ds.dataset
            #   like this: ds.dataset(fpaths, filesystem="s3://my-bucket/", ...)
            # - Passing a SubTreeFileSystem, with s3://my-bucket prefix removed from file paths.
            #   We use this option to be able to pass proxy_options and anonymous options to Arrow.
            #   The endpoint_override option could be passed via the s3 URL as a query option if we wanted
            bucket_name = parsed_url.netloc
            prefix = "s3://" + bucket_name + "/"
            fpaths = [f[len(prefix) :] for f in fpaths]  # remove prefix from file paths
            filesystem = get_s3_subtree_fs(
                bucket_name, region=getfs().region, storage_options=storage_options
            )
        else:
            filesystem = getfs()
        # Presumably the work is partitioned more or less equally among ranks,
        # and we are mostly (or just) reading metadata, so we assign two IO
        # threads to every rank
        pa.set_io_thread_count(4)
        pa.set_cpu_count(4)
        # Use dataset scanner API to get exact row counts when
        # filter is applied. Arrow will try to calculate this by
        # by reading only the file's metadata, and if it needs to
        # access data it will read as less as possible (only the
        # required columns and only subset of row groups if possible)
        dataset_ = ds.dataset(
            fpaths,
            filesystem=filesystem,
            partitioning=ds.partitioning(flavor="hive"),
        )
        for piece, frag in zip(pieces[start:end], dataset_.get_fragments()):
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
            if validate_schema:
                file_schema = frag.metadata.schema.to_arrow_schema()
                if dataset_schema != file_schema:  # pragma: no cover
                    # this is the same error message that pyarrow shows
                    print(
                        "Schema in {!s} was different. \n{!s}\n\nvs\n\n{!s}".format(
                            piece, file_schema, dataset_schema
                        )
                    )
                    valid = False
                    break

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
For best performance the number of row groups should be greater than the number of workers ({bodo.get_size()})
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
                if expr_filters is not None:
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

    # When we pass in a path instead of a filename or list of files to
    # pq.ParquetDataset, the path of the pieces of the resulting dataset
    # object may not contain the prefix such as hdfs://localhost:9000,
    # so we need to store the information in some way since the
    # code that uses this dataset object needs to know this information.
    # Therefore, in case this prefix is not present, we add it
    # to the dataset object as a new attribute (_prefix).

    # Only hdfs seems to be affected based on:
    # https://github.com/apache/arrow/blob/ab307365198d5724a0b3331a05704d0fe4bcd710/python/pyarrow/parquet.py#L50
    # which is called in pq.ParquetDataset.__init__
    # https://github.com/apache/arrow/blob/ab307365198d5724a0b3331a05704d0fe4bcd710/python/pyarrow/parquet.py#L1235
    dataset._prefix = ""
    if protocol in {"hdfs", "abfs", "abfss"}:  # pragma: no cover
        # Compute the prefix that is expected to be there in the
        # path of the pieces
        prefix = f"{protocol}://{parsed_url.netloc}"
        # As a heuristic, we only check if the first piece is missing
        # this prefix
        if len(dataset.pieces) > 0:
            piece = dataset.pieces[0]
            # If it is missing, then assign it to _prefix attribute of
            # the dataset object, else set _prefix to an empty string
            if not piece.path.startswith(prefix):
                dataset._prefix = prefix

    if read_categories:
        _add_categories_to_pq_dataset(dataset)

    if get_row_counts:
        ev.finalize()
    return dataset


def get_scanner_batches(
    fpaths,
    expr_filters,
    selected_fields,
    avg_num_pieces,
    is_parallel,
    storage_options,
    region,
    prefix,  # examples: "hdfs://localhost:9000", "abfs://demo@bododemo.dfs.core.windows.net"
    str_as_dict_cols,
):
    """return RecordBatchReader for dataset of 'fpaths' that contain the rows
    that match expr_filters (or all rows if expr_filters is None). Only project the
    fields in selected_fields"""
    import pyarrow as pa

    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count == 0:
        cpu_count = 2
    default_threads = min(4, cpu_count)
    max_threads = min(16, cpu_count)
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
    if fpaths[0].startswith("s3://"):
        # see comment in get_parquet_dataset about calling ds.dataset with s3
        bucket_name = urlparse(fpaths[0]).netloc
        prefix = "s3://" + bucket_name + "/"
        fpaths = [f[len(prefix) :] for f in fpaths]  # remove prefix from file paths
        filesystem = get_s3_subtree_fs(
            bucket_name, region=region, storage_options=storage_options
        )
    elif prefix and prefix.startswith(("hdfs", "abfs", "abfss")):  # pragma: no cover
        # unlike s3, hdfs file paths already have their prefix removed
        filesystem = get_hdfs_fs(prefix + fpaths[0])
    elif fpaths[0].startswith(("gcs", "gs")):
        # TODO pass storage_options to GCSFileSystem
        import gcsfs

        filesystem = gcsfs.GCSFileSystem(token=None)
    else:
        filesystem = None
    # We only support hive partitioning right now and will need to change this
    # if we also support directory partitioning
    pq_format = ds.ParquetFileFormat(dictionary_columns=str_as_dict_cols)
    dataset = ds.dataset(
        fpaths,
        filesystem=filesystem,
        partitioning=ds.partitioning(flavor="hive"),
        format=pq_format,
    )
    col_names = dataset.schema.names
    selected_names = [col_names[field_num] for field_num in selected_fields]
    # TODO: use threads only for s3?
    reader = dataset.scanner(
        columns=selected_names, filter=expr_filters, use_threads=True
    ).to_reader()
    return dataset, reader


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

    pa_schema = pq_dataset.schema.to_arrow_schema()
    cat_col_names = [
        c
        for c in pa_schema.names
        if isinstance(pa_schema.field(c).type, pa.DictionaryType)
    ]

    # avoid more work if no categorical columns
    if len(cat_col_names) == 0:
        pq_dataset._category_info = {}
        return

    comm = MPI.COMM_WORLD
    if bodo.get_rank() == 0:
        try:
            # read categorical values from first row group of first file
            pf = pq_dataset.pieces[0].open()
            rg = pf.read_row_group(0, cat_col_names)
            # NOTE: assuming DictionaryArray has only one chunk
            category_info = {
                c: tuple(rg.column(c).chunk(0).dictionary.to_pylist())
                for c in cat_col_names
            }
            del pf, rg  # release I/O resources ASAP
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
        # ignore RangeIndex metadata for multi-part datasets
        # TODO: check what pandas/pyarrow does
        if not isinstance(index_col, str) and (
            not isinstance(index_col, dict) or num_pieces != 1
        ):
            index_col = None

        for col_dict in pandas_metadata["columns"]:
            col_name = col_dict["name"]
            if col_dict["pandas_type"].startswith("int") and col_name is not None:
                if col_dict["numpy_type"].startswith("Int"):
                    nullable_from_metadata[col_name] = True
                else:
                    nullable_from_metadata[col_name] = False
    return index_col, nullable_from_metadata


def determine_str_as_dict_columns(pq_dataset, pa_schema):
    """Determine which string columns should be read by Arrow as dictionary
    encoded arrays, based on this heuristic:
    # calculating the ratio of total_uncompressed_size of the column vs number
    # of values.
    # If the ratio is less than READ_STR_AS_DICT_THRESHOLD we read as DICT"""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    str_columns = []
    for col_name in pa_schema.names:
        field = pa_schema.field(col_name)
        if field.type == pa.string():
            str_columns.append((col_name, pa_schema.get_field_index(col_name)))
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
        metadata = piece.get_metadata()
        for i in range(metadata.num_row_groups):
            for j, (_, idx) in enumerate(str_columns):
                total_uncompressed_sizes[j] += (
                    metadata.row_group(i).column(idx).total_uncompressed_size
                )
        num_rows = metadata.num_rows
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
    file_name, selected_columns, storage_options=None, input_file_name_col=None
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

    # not using 'partition_names' since the order may not match 'levels'
    partition_names = (
        []
        if pq_dataset.partitions is None
        else [
            pq_dataset.partitions.levels[i].name
            for i in range(len(pq_dataset.partitions.partition_names))
        ]
    )
    pa_schema = pq_dataset.schema.to_arrow_schema()
    num_pieces = len(pq_dataset.pieces)

    str_as_dict = determine_str_as_dict_columns(pq_dataset, pa_schema)

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
        col_names += partition_names
        col_types_total += [
            _get_partition_cat_dtype(pq_dataset.partitions.levels[i])
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


def _get_partition_cat_dtype(part_set):
    """get categorical dtype for Parquet partition set"""
    # using 'dictionary' instead of 'keys' attribute since 'keys' may not have the
    # right data type (e.g. string instead of int64)
    S = part_set.dictionary.to_pandas()
    elem_type = bodo.typeof(S).dtype
    cat_dtype = PDCategoricalDtype(tuple(S), elem_type, False)
    return CategoricalArrayType(cat_dtype)


_pq_read = types.ExternalFunction(
    "pq_read",
    table_type(
        read_parquet_fpath_type,
        types.boolean,
        types.voidptr,
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
):
    """
    Call C++ parquet write function
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
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
            ],
        )
        fn_tp = cgutils.get_or_insert_function(builder.module, fnty, name="pq_write")
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    return (
        types.void(
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
        ),
        codegen,
    )
