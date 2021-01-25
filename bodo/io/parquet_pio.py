# Copyright (C) 2019 Bodo Inc. All rights reserved.
import os
from collections import defaultdict

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
from numba.core import ir, types
from numba.core.imputils import impl_ret_new_ref
from numba.core.ir_utils import (
    compile_to_numba_ir,
    get_definition,
    guard,
    mk_unique_var,
    replace_arg_nodes,
)
from numba.core.typing import signature
from numba.core.typing.templates import AbstractTemplate, infer_global
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
from bodo.hiframes.datetime_date_ext import (
    datetime_date_array_type,
    datetime_date_type,
)
from bodo.hiframes.pd_series_ext import _get_series_array_type
from bodo.io import parquet_cpp
from bodo.io.fs_io import (
    abfs_is_directory,
    abfs_list_dir_fnames,
    directory_of_files_common_filter,
    get_hdfs_fs,
    get_s3_fs,
    hdfs_is_directory,
    hdfs_list_dir_fnames,
    s3_is_directory,
    s3_list_dir_fnames,
)
from bodo.libs.array import _lower_info_to_array_numpy, array_info_type
from bodo.libs.array_item_arr_ext import (
    ArrayItemArrayPayloadType,
    ArrayItemArrayType,
    define_array_item_dtor,
    offset_type,
)
from bodo.libs.bool_arr_ext import BooleanArrayType, boolean_array
from bodo.libs.decimal_arr_ext import Decimal128Type, DecimalArrayType
from bodo.libs.distributed_api import get_end, get_start
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_arr_ext import char_arr_type, string_array_type
from bodo.libs.str_ext import string_type, unicode_to_utf8
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.transforms import distributed_pass
from bodo.utils.transform import get_const_value
from bodo.utils.typing import BodoError, FileInfo, get_overload_const_str
from bodo.utils.utils import (
    check_and_propagate_cpp_exception,
    is_null_pointer,
    sanitize_varname,
    unliteral_all,
)

# read Arrow Int columns as nullable int array (IntegerArrayType)
use_nullable_int_arr = True


import pyarrow.parquet as pq


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


def ParquetManifest_visit_level(self, level, base_path, part_keys):
    fs = self.filesystem

    _, directories, files = next(fs.walk(base_path))

    filtered_files = []
    for path in files:
        if path == "":  # Bodo change
            continue
        full_path = self.pathsep.join((base_path, path))
        if path.endswith("_common_metadata"):
            self.common_metadata_path = full_path
        elif path.endswith("_metadata"):
            self.metadata_path = full_path
        elif self._should_silently_exclude(path):
            continue
        else:
            filtered_files.append(full_path)

    # ARROW-1079: Filter out "private" directories starting with underscore
    filtered_directories = [
        self.pathsep.join((base_path, x))
        for x in directories
        if not pq._is_private_directory(x)
    ]

    filtered_files.sort()
    filtered_directories.sort()

    if len(filtered_files) > 0 and len(filtered_directories) > 0:
        raise ValueError(
            "Found files in an intermediate " "directory: {}".format(base_path)
        )
    elif len(filtered_directories) > 0:
        self._visit_directories(level, filtered_directories, part_keys)
    else:
        self._push_pieces(filtered_files, part_keys)


pq.ParquetManifest._visit_level = ParquetManifest_visit_level


def read_parquet():  # pragma: no cover
    return 0


def read_parquet_str():  # pragma: no cover
    return 0


def read_parquet_list_str():  # pragma: no cover
    return 0


def read_parquet_array_item():  # pragma: no cover
    return 0


def read_parquet_arrow_array():  # pragma: no cover
    return 0


def get_column_size_parquet():  # pragma: no cover
    return 0


class ParquetFileInfo(FileInfo):
    """FileInfo object passed to ForceLiteralArg for
    file name arguments that refer to a parquet dataset"""

    def __init__(self, columns):
        self.columns = columns  # columns to select from parquet dataset

    def get_schema(self, fname):
        return parquet_file_schema(fname, selected_columns=self.columns)


class ParquetHandler:
    """analyze and transform parquet IO calls"""

    def __init__(self, func_ir, typingctx, args, _locals, _reverse_copies):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.args = args
        self.locals = _locals
        self.reverse_copies = _reverse_copies

    def gen_parquet_read(self, file_name, lhs, columns):
        scope = lhs.scope
        loc = lhs.loc

        table_types = None
        # lhs is temporary and will possibly be assigned to user variable
        assert lhs.name.startswith("$")
        if (
            lhs.name in self.reverse_copies
            and self.reverse_copies[lhs.name] in self.locals
        ):
            table_types = self.locals[self.reverse_copies[lhs.name]]
            self.locals.pop(self.reverse_copies[lhs.name])

        convert_types = {}
        # user-specified type conversion
        if (
            lhs.name in self.reverse_copies
            and (self.reverse_copies[lhs.name] + ":convert") in self.locals
        ):
            convert_types = self.locals[self.reverse_copies[lhs.name] + ":convert"]
            self.locals.pop(self.reverse_copies[lhs.name] + ":convert")

        if table_types is None:
            msg = (
                "Parquet schema not available. Either path "
                "argument should be constant for Bodo to look at the file "
                "at compile time or schema should be provided."
            )
            file_name_str = get_const_value(
                file_name,
                self.func_ir,
                msg,
                arg_types=self.args,
                file_info=ParquetFileInfo(columns),
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
                        col_nb_fields,
                        null_col_map,
                        partition_names,
                    ) = typ.schema
                    got_schema = True
            if not got_schema:
                (
                    col_names,
                    col_types,
                    index_col,
                    col_indices,
                    col_nb_fields,
                    null_col_map,
                    partition_names,
                ) = parquet_file_schema(file_name_str, columns)
        else:
            col_names_total = list(table_types.keys())
            col_types_total = [t for t in table_types.values()]
            index_col = "index" if "index" in col_names_total else None
            col_nb_fields_total = [
                compute_number_of_fields(c_typ) for c_typ in col_types_total
            ]
            col_indices_total = compute_column_index(col_nb_fields_total)
            # TODO: allow specifying types of only selected columns
            if columns is None:
                selected_columns = col_names_total
            else:
                selected_columns = columns
            # store tuples of (real_column_index, index of first column in parquet file)
            col_indices = []
            for c in selected_columns:
                real_col_idx = col_names_total.index(c)
                col_indices.append((real_col_idx, col_indices_total[real_col_idx]))
            col_types = [
                col_types_total[col_names_total.index(c)] for c in selected_columns
            ]
            col_nb_fields = [
                col_nb_fields_total[col_names_total.index(c)] for c in selected_columns
            ]
            col_names = selected_columns
            index_col = index_col if index_col in col_names else None
            # Initialize null_col_map. See parquet_file_schema for definition.
            null_col_map = [False] * len(col_names)
            partition_names = None

        # HACK convert types using decorator for int columns with NaN
        for i, c in enumerate(col_names):
            if c in convert_types:
                col_types[i] = convert_types[c]

        data_arrs = [ir.Var(scope, mk_unique_var(c), loc) for c in col_names]

        nodes = [
            bodo.ir.parquet_ext.ParquetReader(
                file_name,
                lhs.name,
                col_names,
                col_indices,
                col_nb_fields,
                col_types,
                data_arrs,
                loc,
                null_col_map,
                partition_names,
            )
        ]

        return col_names, data_arrs, index_col, nodes


def pq_distributed_run(pq_node, array_dists, typemap, calltypes, typingctx, targetctx):
    """lower ParquetReader into regular Numba nodes. Generates code for Parquet
    data read.
    """

    n_cols = len(pq_node.out_vars)
    filter_vars = []
    extra_args = ""
    filter_str = "None"
    if pq_node.filters:
        # handle predicate pushdown variables that need to be passed to C++
        pred_vars = [v[2] for predicate_list in pq_node.filters for v in predicate_list]
        # variables may be repeated due to distribution of Or over And in predicates, so
        # remove duplicates. Cannot use ir.Var objects in set directly.
        var_set = set()
        for var in pred_vars:
            if var.name not in var_set:
                filter_vars.append(var)
            var_set.add(var.name)
        vararg_map = {v.name: f"f{i}" for i, v in enumerate(filter_vars)}
        filter_str = "[{}]".format(
            ", ".join(
                "[{}]".format(
                    ", ".join(
                        f"('{v[0]}', '{v[1]}', {vararg_map[v[2].name]})"
                        for v in predicate
                    )
                )
                for predicate in pq_node.filters
            )
        )
        extra_args = ", ".join(vararg_map.values())

    # get column variables and their sizes
    arg_names = ", ".join("out" + str(i) for i in range(2 * n_cols))
    func_text = f"def pq_impl(fname, {extra_args}):\n"
    func_text += f"    ({arg_names},) = _pq_reader_py(fname, {extra_args})\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    pq_impl = loc_vars["pq_impl"]

    # parallel columns
    parallel = []
    if array_dists is not None:
        parallel = [
            c
            for c, v in zip(pq_node.col_names, pq_node.out_vars)
            if array_dists[v.name]
            in (
                distributed_pass.Distribution.OneD,
                distributed_pass.Distribution.OneD_Var,
            )
        ]

    pq_reader_py = _gen_pq_reader_py(
        pq_node.col_names,
        pq_node.col_indices,
        pq_node.col_nb_fields,
        pq_node.out_types,
        pq_node.null_col_map,
        pq_node.partition_names,
        filter_str,
        extra_args,
        typingctx,
        targetctx,
        parallel,
    )
    arg_types = (string_type,) + tuple(typemap[v.name] for v in filter_vars)
    f_block = compile_to_numba_ir(
        pq_impl,
        {"_pq_reader_py": pq_reader_py},
        typingctx,
        arg_types,
        typemap,
        calltypes,
    ).blocks.popitem()[1]
    replace_arg_nodes(f_block, [pq_node.file_name] + filter_vars)
    nodes = f_block.body[:-3]

    for i in range(n_cols):
        nodes[2 * (i - n_cols)].target = pq_node.out_vars[i]

    return nodes


distributed_pass.distributed_run_extensions[
    bodo.ir.parquet_ext.ParquetReader
] = pq_distributed_run


def get_filters_pyobject(filters, vars):  # pragma: no cover
    pass


@overload(get_filters_pyobject, no_unliteral=True)
def overload_get_filters_pyobject(filter_str, var_tup):
    """generate a pyobject for filter expression to pass to C++"""
    filter_str_val = get_overload_const_str(filter_str)
    var_unpack = ", ".join(f"f{i}" for i in range(len(var_tup)))
    func_text = "def impl(filter_str, var_tup):\n"
    if len(var_tup):
        func_text += f"  {var_unpack}, = var_tup\n"
    func_text += "  with numba.objmode(filters_py='parquet_predicate_type'):\n"
    func_text += f"    filters_py = {filter_str_val}\n"
    func_text += "  return filters_py\n"
    loc_vars = {}
    exec(func_text, {"numba": numba}, loc_vars)
    return loc_vars["impl"]


def _gen_pq_reader_py(
    col_names,
    col_indices,
    col_nb_fields,
    out_types,
    null_col_map,
    partition_names,
    filter_str,
    extra_args,
    typingctx,
    targetctx,
    parallel,
):
    if len(parallel) > 0:
        # in parallel read, we assume all columns are parallel
        assert col_names == parallel
    is_parallel = len(parallel) > 0
    comma = "," if extra_args else ""
    func_text = f"def pq_reader_py(fname,{extra_args}):\n"
    # if it's an s3 url, get the region and pass it into the c++ code
    func_text += "  bucket_region = bodo.io.fs_io.get_s3_bucket_region_njit(fname)\n"
    func_text += (
        f'  filters = get_filters_pyobject("{filter_str}", ({extra_args}{comma}))\n'
    )
    # open a DatasetReader, which is a C++ object defined in _parquet.cpp that
    # contains file readers for the files from which this process needs to read,
    # and other information to read this process' chunk
    func_text += "  ds_reader = get_dataset_reader(unicode_to_utf8(fname), {}, unicode_to_utf8(bucket_region), filters)\n".format(
        is_parallel
    )
    # Check if there was an error in the C++ code. If so, raise it.
    func_text += "  check_and_propagate_cpp_exception()\n"
    func_text += "  if is_null_pointer(ds_reader):\n"
    func_text += "    raise ValueError('Error reading Parquet dataset')\n"

    local_types = {}
    if partition_names is None:
        partition_names = []
        first_partition_name = None
    else:
        first_partition_name = partition_names[0]
    for c_name, (real_col_ind, col_ind), col_siz, c_typ, is_all_null in zip(
        col_names, col_indices, col_nb_fields, out_types, null_col_map
    ):
        if c_name == first_partition_name:
            # partition columns appear last and are handled below
            break
        ret_type, func_text = gen_column_read(
            func_text,
            c_name,
            real_col_ind,
            col_ind,
            col_siz,
            c_typ,
            c_name in parallel,
            is_all_null,
        )
        if ret_type != None:
            local_types[ret_type] = c_typ

    for part_col_idx, part_name in enumerate(partition_names):
        try:
            col_idx = col_names.index(part_name)
        except ValueError:
            # this partition column has not been selected for read
            continue
        part_col_type = out_types[col_idx].dtype
        cat_dtype, cat_dtype_name, func_text = gen_column_partition(
            func_text, part_col_idx, part_name, part_col_type
        )
        local_types[cat_dtype_name] = cat_dtype

    func_text += "  del_dataset_reader(ds_reader)\n"
    func_text += "  return ({},)\n".format(
        ", ".join("{0}, {0}_size".format(sanitize_varname(c)) for c in col_names)
    )
    loc_vars = {}
    glbs = {
        "get_dataset_reader": _get_dataset_reader,
        "check_and_propagate_cpp_exception": check_and_propagate_cpp_exception,
        "is_null_pointer": is_null_pointer,
        "del_dataset_reader": _del_dataset_reader,
        "get_column_size_parquet": get_column_size_parquet,
        "read_parquet": read_parquet,
        "read_parquet_str": read_parquet_str,
        "read_parquet_list_str": read_parquet_list_str,
        "read_parquet_array_item": read_parquet_array_item,
        "read_parquet_arrow_array": read_parquet_arrow_array,
        "pq_gen_partition_column": _pq_gen_partition_column,
        "unicode_to_utf8": unicode_to_utf8,
        "get_filters_pyobject": get_filters_pyobject,
        "NS_DTYPE": np.dtype("M8[ns]"),
        "np": np,
        "bodo": bodo,
    }
    glbs.update(local_types)

    exec(func_text, glbs, loc_vars)
    pq_reader_py = loc_vars["pq_reader_py"]

    jit_func = numba.njit(pq_reader_py, no_cpython_wrapper=True)
    return jit_func


def gen_column_read(
    func_text, cname, c_ind_real, c_ind, c_siz, c_type, is_parallel, is_all_null
):
    cname = sanitize_varname(cname)
    # handle size variables
    # get_column_size_parquet returns local size of column (number of rows
    # that this process is going to read)
    func_text += "  {}_size = get_column_size_parquet(ds_reader, {})\n".format(
        cname, c_ind
    )
    # Check if there was an error in the C++ code. If so, raise it.
    func_text += "  bodo.utils.utils.check_and_propagate_cpp_exception()\n"
    alloc_size = "{}_size".format(cname)

    ret_type = None
    if is_all_null:
        # if the column is made up of all nulls, initialize it as a string array of NaNs.
        func_text += "  {0} = bodo.libs.str_arr_ext.pre_alloc_string_array({0}_size, 0)\n".format(
            cname
        )
        func_text += "  bodo.libs.str_arr_ext.set_null_bits_to_value({}, 0)\n".format(
            cname
        )
    elif c_type == string_array_type:
        # pass size for easier allocation and distributed analysis
        func_text += "  {} = read_parquet_str(ds_reader, {}, {}, {}_size)\n".format(
            cname, c_ind_real, c_ind, cname
        )
    elif c_type == ArrayItemArrayType(string_array_type):
        # TODO does not support null strings?
        # pass size for easier allocation and distributed analysis
        func_text += (
            "  {} = read_parquet_list_str(ds_reader, {}, {}, {}_size)\n".format(
                cname, c_ind_real, c_ind, cname
            )
        )
    elif isinstance(c_type, ArrayItemArrayType):
        # Force generalized path for all ArrayItemArrayType(...) (see FIXME below)
        if not isinstance(c_type.dtype, types.Float):
            ret_type = "nested_type{}".format(c_ind)
            func_text += (
                "  {} = read_parquet_arrow_array(ds_reader, {}, {}, {}, {})\n".format(
                    cname, c_ind_real, c_ind, c_siz, ret_type
                )
            )
        else:
            # FIXME This code path does not support nullable items,
            # for example: ArrayItemArray(IntegerArray(int64))
            # The main gap is in ReadParquetArrayItemInfer and pq_read_array_item_lower
            # Right now we use it only for the float where we do not need nullable items.
            elem_typ = c_type.dtype.dtype
            func_text += "  {} = read_parquet_array_item(ds_reader, {}, {}, {}_size, np.int32({}), {})\n".format(
                cname,
                c_ind_real,
                c_ind,
                cname,
                bodo.utils.utils.numba_to_c_type(elem_typ),
                get_element_type(elem_typ),
            )
    elif isinstance(c_type, StructArrayType):
        ret_type = "nested_type{}".format(c_ind)
        func_text += (
            "  {} = read_parquet_arrow_array(ds_reader, {}, {}, {}, {})\n".format(
                cname, c_ind_real, c_ind, c_siz, ret_type
            )
        )
    else:
        el_type = get_element_type(c_type.dtype)
        func_text += _gen_alloc(c_type, cname, alloc_size, el_type)
        func_text += (
            "  status = read_parquet(ds_reader, {}, {}, {}, np.int32({}))\n".format(
                c_ind_real,
                c_ind,
                cname,
                bodo.utils.utils.numba_to_c_type(c_type.dtype),
            )
        )

    return ret_type, func_text


def gen_column_partition(
    func_text,
    part_col_idx,
    partition_name,
    partition_type,
):
    cname = sanitize_varname(partition_name)
    cat_dtype = pd.CategoricalDtype(partition_type.categories, partition_type.ordered)
    cat_dtype_name = "part_pd_cat_dtype_{}".format(part_col_idx)

    # handle size variables
    # get_column_size_parquet returns local size of column (number of rows
    # that this process is going to read)
    # XXX partition is not part of the parquet file. Assume it is same
    # length as column 0?
    func_text += "  {}_size = get_column_size_parquet(ds_reader, 0)\n".format(cname)
    # Check if there was an error in the C++ code. If so, raise it.
    func_text += "  bodo.utils.utils.check_and_propagate_cpp_exception()\n"
    func_text += "  {} = bodo.hiframes.pd_categorical_ext.alloc_categorical_array({}_size, {})\n".format(
        cname, cname, cat_dtype_name
    )
    cat_int_dtype = bodo.hiframes.pd_categorical_ext.get_categories_int_type(
        partition_type
    )
    func_text += "  pq_gen_partition_column(ds_reader, {}, {}.codes.ctypes, np.int32({}))\n".format(
        part_col_idx, cname, bodo.utils.utils.numba_to_c_type(cat_int_dtype)
    )
    func_text += "  bodo.utils.utils.check_and_propagate_cpp_exception()\n"
    return cat_dtype, cat_dtype_name, func_text


def _gen_alloc(c_type, cname, alloc_size, el_type):
    if isinstance(c_type, IntegerArrayType):
        return "  {0} = bodo.libs.int_arr_ext.alloc_int_array({1}, {2})\n".format(
            cname, alloc_size, el_type
        )
    if c_type == boolean_array:
        return "  {0} = bodo.libs.bool_arr_ext.alloc_bool_array({1})\n".format(
            cname, alloc_size
        )
    if isinstance(c_type, DecimalArrayType):
        return "  {0} = bodo.libs.decimal_arr_ext.alloc_decimal_array({1}, {2}, {3})\n".format(
            cname, alloc_size, c_type.precision, c_type.scale
        )
    if c_type == datetime_date_array_type:
        return "  {0} = bodo.hiframes.datetime_date_ext.alloc_datetime_date_array({1})\n".format(
            cname, alloc_size
        )
    return "  {} = np.empty({}, dtype={})\n".format(cname, alloc_size, el_type)


def get_element_type(dtype):
    """get dtype string to pass to empty() allocations"""
    if dtype == types.NPDatetime("ns"):
        # NS_DTYPE has to be defined in function globals
        return "NS_DTYPE"

    if isinstance(dtype, Decimal128Type):
        return "int128"

    if dtype == datetime_date_type:
        return "np.int32"

    out = repr(dtype)
    if out == "bool":  # fix bool string
        out = "bool_"

    return "np." + out


def compute_number_of_fields(numba_typ):
    """For data structures like structs, the data is stored on several columns
    in parquet. This has to be determined beforehand."""
    if isinstance(numba_typ, ArrayItemArrayType):
        return compute_number_of_fields(numba_typ.dtype)
    if isinstance(numba_typ, StructArrayType):
        return sum([compute_number_of_fields(e_ent) for e_ent in numba_typ.data])
    return 1


def compute_column_index(list_siz):
    """The columns have a number of fields. From the list of number of fields the
    list of indices are computed."""
    sum_index = 0
    list_cumsum = []
    for e_siz in list_siz:
        list_cumsum.append(sum_index)
        sum_index += e_siz
    return list_cumsum


def _get_numba_typ_from_pa_typ(pa_typ, is_index, nullable_from_metadata):
    import pyarrow as pa

    if isinstance(pa_typ.type, pa.ListType):
        return ArrayItemArrayType(
            # nullable_from_metadata is only used for non-nested Int arrays
            _get_numba_typ_from_pa_typ(
                pa_typ.type.value_field, is_index, nullable_from_metadata
            )
        )

    if isinstance(pa_typ.type, pa.StructType):
        child_types = []
        field_names = []
        for field in pa_typ.flatten():
            field_names.append(field.name.split(".")[-1])
            child_types.append(
                _get_numba_typ_from_pa_typ(field, is_index, nullable_from_metadata)
            )
        return StructArrayType(tuple(child_types), tuple(field_names))

    # Decimal128Array type
    if isinstance(pa_typ.type, pa.Decimal128Type):
        return DecimalArrayType(pa_typ.type.precision, pa_typ.type.scale)

    _typ_map = {
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

    if pa_typ.type not in _typ_map:
        raise BodoError("Arrow data type {} not supported yet".format(pa_typ.type))
    dtype = _typ_map[pa_typ.type]

    if dtype == datetime_date_type:
        return datetime_date_array_type

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

    return arr_typ


def merge_subdatasets(subdatasets, get_row_counts):
    """Merge datasets of type pyarrow.parquet.ParquetDataset into one.
    Each dataset contains a subset of pieces (parquet files) of the same dataset.
    This function will check that the schema of all datasets is the same.
    If get_row_counts is True, pieces must have '_bodo_num_rows' attribute
    to indicate the number of rows of each piece, and the total number
    of rows will be saved to the resulting dataset in '_bodo_total_rows'
    attribute.
    NOTE: this will modify the first dataset in the list
    """
    dataset = None
    for subdataset in subdatasets:
        if subdataset is None:
            continue
        if isinstance(subdataset, Exception):
            e = subdataset
            raise e
        if dataset is None:
            # This is the first valid dataset in the list. We use this object
            # to generate the result
            dataset = subdataset
            dataset._bodo_total_rows = 0
        else:
            # add this dataset's pieces to the result
            dataset.pieces += subdataset.pieces
        if get_row_counts:
            for piece in subdataset.pieces:
                dataset._bodo_total_rows += piece._bodo_num_rows
        if dataset.schema != subdataset.schema:
            raise BodoError("Files in directory have non-matching schema")
    if dataset is None:
        raise BodoError("No parquet files found in directory")
    return dataset


def is_directory_parallel(fpath):
    """Determines whether a path is a directory. Is meant to be called by
    all ranks. Rank 0 will do the check and communicate the result."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    isdir = None
    if bodo.get_rank() == 0:
        try:
            if fpath.startswith("s3://"):
                isdir = s3_is_directory(get_s3_fs(), fpath)
            elif fpath.startswith("hdfs://"):  # pragma: no cover
                _, isdir = hdfs_is_directory(fpath)
            # TODO merge with hdfs when new pyarrow API works with abfs
            elif fpath.startswith("abfs://") or fpath.startswith(
                "abfss://"
            ):  # pragma: no cover
                _, isdir = abfs_is_directory(fpath)
            else:
                isdir = os.path.isdir(fpath)
        except Exception as e:
            isdir = e
    isdir = comm.bcast(isdir)
    if isinstance(isdir, Exception):
        e = isdir
        raise e
    return isdir


def is_nested_parallel(fpath, file_names):
    """Returns True if one of the file names in file_names is a directory"""

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    is_nested = False
    if bodo.get_rank() == 0:
        try:
            if fpath.startswith("s3://"):
                fs = get_s3_fs()
                for fname in file_names:
                    if s3_is_directory(fs, fname):
                        is_nested = True
                        break
            else:
                if fpath.startswith("hdfs://"):  # pragma: no cover
                    is_dir_func = hdfs_is_directory
                # TODO merge with hdfs when new pyarrow API works with abfs
                elif fpath.startswith("abfs://") or fpath.startswith(
                    "abfss://"
                ):  # pragma: no cover
                    is_dir_func = abfs_is_directory
                else:
                    is_dir_func = os.path.isdir
                for fname in file_names:
                    if is_dir_func(fname):
                        is_nested = True
                        break
        except Exception as e:
            is_nested = e
    is_nested = comm.bcast(is_nested)
    if isinstance(is_nested, Exception):
        e = is_nested
        raise e
    return is_nested


def get_filenames_parallel(path, filter_func):
    """Get sorted list of file names in directory 'path'. Filter files based
    on the filter function 'filter_func'.
    Is meant to be called by all ranks. Rank 0 will get the list and
    communicate the result."""

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    file_names = None
    path = path.rstrip("/")  # remove any trailing / from path
    if bodo.get_rank() == 0:
        try:
            if path.startswith("s3://"):
                file_names = s3_list_dir_fnames(get_s3_fs(), path)
            elif path.startswith("hdfs://"):  # pragma: no cover
                _, file_names = hdfs_list_dir_fnames(path)
            # TODO merge with hdfs when new pyarrow API works with abfs
            elif path.startswith("abfs://") or path.startswith(
                "abfss://"
            ):  # pragma: no cover
                _, file_names = abfs_list_dir_fnames(path)
            else:
                file_names = os.listdir(path)
            # pq.ParquetDataset() needs the full path for each file
            file_names = [
                (path + "/" + f) for f in sorted(filter(filter_func, file_names))
            ]
        except Exception as e:
            file_names = e
    file_names = comm.bcast(file_names)
    if isinstance(file_names, Exception):
        e = file_names
        raise e
    return file_names


def get_parquet_dataset(fpath, parallel, get_row_counts=True, filters=None):
    """get ParquetDataset object for 'fpath' and set the number of total rows as an
    attribute. Also, sets the number of rows per file as an attribute of
    ParquetDatasetPiece objects.
    'filters' are used for predicate pushdown which prunes the unnecessary pieces.
    """

    import pyarrow.parquet as pq
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    fs = []

    def getfs():
        if len(fs) == 1:
            return fs[0]
        if fpath.startswith("s3://"):
            fs.append(get_s3_fs())
        elif (
            fpath.startswith("hdfs://")
            or fpath.startswith("abfs://")
            or fpath.startswith("abfss://")
        ):  # pragma: no cover
            fs.append(get_hdfs_fs(fpath))
        else:
            fs.append(None)
        return fs[0]

    if is_directory_parallel(fpath):
        file_names = get_filenames_parallel(fpath, directory_of_files_common_filter)
        if filters is not None or is_nested_parallel(fpath, file_names):
            # For now, for nested directories (for example partitioned datasets using
            # hive scheme) we let pyarrow.parquet.ParquetDataset
            # extract all the information by passing it the top-level directory.
            # This happens sequentially.
            # Note: ParquetDataset has metadata_nthreads parameter but it
            # seems to use Python threads and not make a difference
            if bodo.get_rank() == 0:
                dataset = pq.ParquetDataset(fpath, filesystem=getfs(), filters=filters)
                dataset._bodo_total_rows = 0
                if get_row_counts:
                    for piece in dataset.pieces:
                        piece._bodo_num_rows = piece.get_metadata().num_rows
                        dataset._bodo_total_rows += piece._bodo_num_rows
                comm.bcast(dataset)
            else:
                dataset = comm.bcast(None)
        else:
            # Note that at this point there could still be files in file_names
            # which aren't valid parquet files, so the division of work between
            # ranks below won't be perfect.
            # We can either do better filtering in `get_filenames_parallel` or
            # do a further subdivision of work after checking which files are
            # parquet files below. But I think this is already good.
            try:
                start = get_start(len(file_names), bodo.get_size(), bodo.get_rank())
                end = get_end(len(file_names), bodo.get_size(), bodo.get_rank())
                # When passing list of files these can only be valid parquet files,
                # so we check each one of them manually.
                # Note that for s3 this is needed anyway (if you call pq.ParquetDataset()
                # with a directory name and the directory contains non-parquet files
                # it will fail). And we are doing this in parallel so it shouldn't
                # be an issue.
                myfiles = []
                for fname in file_names[start:end]:
                    try:
                        pq.ParquetDataset(fname, filesystem=getfs())
                        myfiles.append(fname)
                    except:
                        # this is not a valid parquet file
                        pass
                subdataset = None
                if len(myfiles) > 0:
                    subdataset = pq.ParquetDataset(myfiles, filesystem=getfs())
                    if get_row_counts:
                        for piece in subdataset.pieces:
                            piece._bodo_num_rows = piece.get_metadata().num_rows
            except Exception as e:
                subdataset = e

            subdatasets = comm.allgather(subdataset)
            # merge all the subdatasets into one
            dataset = merge_subdatasets(subdatasets, get_row_counts)
    else:
        if bodo.get_rank() == 0:
            try:
                dataset = pq.ParquetDataset(fpath, filesystem=getfs(), filters=filters)
                if get_row_counts:
                    dataset._bodo_total_rows = 0
                    for piece in dataset.pieces:
                        piece._bodo_num_rows = piece.get_metadata().num_rows
                        dataset._bodo_total_rows += piece._bodo_num_rows
            except Exception as e:
                dataset = e
            comm.bcast(dataset)
        else:
            dataset = comm.bcast(None)

    if isinstance(dataset, Exception):
        e = dataset
        raise e

    return dataset


def parquet_file_schema(file_name, selected_columns):
    """get parquet schema from file using Parquet dataset and Arrow APIs"""
    col_names = []
    col_types = []

    # during compilation we only need the schema and it has to be the same for
    # all processes, so we can set parallel=True to just have rank 0 read
    # the dataset information and broadcast to others
    pq_dataset = get_parquet_dataset(file_name, parallel=True, get_row_counts=False)
    # not using 'partition_names' since the order may not match 'levels'
    partition_names = (
        None
        if pq_dataset.partitions is None
        else [
            pq_dataset.partitions.levels[i].name
            for i in range(len(pq_dataset.partitions.partition_names))
        ]
    )
    pa_schema = pq_dataset.schema.to_arrow_schema()
    num_pieces = len(pq_dataset.pieces)

    # NOTE: use arrow schema instead of the dataset schema to avoid issues with
    # names of list types columns (arrow 0.17.0)
    # col_names is an array that contains all the column's name and
    # index's name if there is one, otherwise "__index__level_0"
    # If there is no index at all, col_names will not include anything.
    col_names = pa_schema.names

    # Create a bit map specifying if a column is made up of all nulls.
    # We use the string type for such columns.
    # But this case is handled separately later on (using a pre-allocated string array),
    # so we need to know which columns need to be handled this way.
    null_col_map = [pa_schema.field(c).type == null() for c in range(len(col_names))]

    # find pandas index column if any
    # TODO: other pandas metadata like dtypes needed?
    # https://pandas.pydata.org/pandas-docs/stable/development/developer.html
    index_col = None
    # column_name -> is_nullable (or None if unknown)
    nullable_from_metadata = defaultdict(lambda: None)
    key = b"pandas"
    if pa_schema.metadata is not None and key in pa_schema.metadata:
        import json

        pandas_metadata = json.loads(pa_schema.metadata[key].decode("utf8"))
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
            if col_dict["pandas_type"].startswith("int"):
                if col_dict["numpy_type"].startswith("Int"):
                    nullable_from_metadata[col_name] = True
                else:
                    nullable_from_metadata[col_name] = False

    col_types_total = [
        _get_numba_typ_from_pa_typ(
            pa_schema.field(c), c == index_col, nullable_from_metadata[c]
        )
        for c in col_names
    ]

    # add partition column data types if any
    if partition_names:
        col_names += partition_names
        null_col_map += [False] * len(partition_names)
        col_types_total += [
            _get_partition_cat_dtype(pq_dataset.partitions.levels[i])
            for i in range(len(partition_names))
        ]

    col_nb_fields_total = [compute_number_of_fields(typ) for typ in col_types_total]
    col_indices_total = compute_column_index(col_nb_fields_total)
    # The numbering of the columns in pandas does not match the numbering in parquet.
    # That is a column of structs {"A": 1, "B": 3} is 1 column in pandas but 2 columns
    # in parquet. Fortunately, those 2 columns are consecutive in parquet.
    # Thus we need to compute the shifts in order to match them.
    # The col_nb_fields is the total number of fields in the column (so 2 in above example).
    # In parquet it is named "column size".
    # The col_indices is for each bodo column the first index in parquet.

    # if no selected columns, set it to all of them.
    if selected_columns is None:
        selected_columns = col_names

    # make sure selected columns are in the schema
    for c in selected_columns:
        if c not in col_names:
            raise BodoError("Selected column {} not in Parquet file schema".format(c))
    if (
        index_col
        and not isinstance(index_col, dict)
        and index_col not in selected_columns
    ):
        # if index_col is "__index__level_0" or some other name, append it.
        # If the index column is not selected when reading parquet, the index
        # should still be included.
        selected_columns.append(index_col)

    # store tuples of (real_column_index, index of first column in parquet file)
    col_indices = []
    for c in selected_columns:
        real_col_idx = col_names.index(c)
        col_indices.append((real_col_idx, col_indices_total[real_col_idx]))
    col_types = [col_types_total[col_names.index(c)] for c in selected_columns]
    col_nb_fields = [col_nb_fields_total[col_names.index(c)] for c in selected_columns]
    null_col_map = [null_col_map[col_names.index(c)] for c in selected_columns]
    col_names = selected_columns
    # TODO: close file?
    return (
        col_names,
        col_types,
        index_col,
        col_indices,
        col_nb_fields,
        null_col_map,
        partition_names,
    )


def _get_partition_cat_dtype(part_set):
    """get categorical dtype for Parquet partition set"""
    # using 'dictionary' instead of 'keys' attribute since 'keys' may not have the
    # right data type (e.g. string instead of int64)
    S = part_set.dictionary.to_pandas()
    elem_type = bodo.typeof(S).dtype
    cat_dtype = bodo.hiframes.pd_categorical_ext.PDCategoricalDtype(
        tuple(S), elem_type, False
    )
    return bodo.hiframes.pd_categorical_ext.CategoricalArray(cat_dtype)


_get_dataset_reader = types.ExternalFunction(
    "get_dataset_reader",
    types.Opaque("arrow_reader")(
        types.voidptr, types.boolean, types.voidptr, parquet_predicate_type
    ),
)
_del_dataset_reader = types.ExternalFunction(
    "del_dataset_reader", types.void(types.Opaque("arrow_reader"))
)

_pq_gen_partition_column = types.ExternalFunction(
    "pq_gen_partition_column",
    types.void(
        types.Opaque("arrow_reader"),
        types.int64,
        types.voidptr,
        types.int32,
    ),
)
ll.add_symbol("pq_gen_partition_column", parquet_cpp.pq_gen_partition_column)


@infer_global(get_column_size_parquet)
class SizeParquetInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(types.intp, args[0], types.unliteral(args[1]))


@infer_global(read_parquet)
class ReadParquetInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 5
        if args[3] == types.intp:  # string read call, returns string array
            return signature(string_array_type, *unliteral_all(args))
        return signature(types.int64, *unliteral_all(args))


@infer_global(read_parquet_str)
class ReadParquetStrInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 4
        return signature(string_array_type, *unliteral_all(args))


@infer_global(read_parquet_list_str)
class ReadParquetListStrInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 4
        return signature(ArrayItemArrayType(string_array_type), *unliteral_all(args))


@infer_global(read_parquet_array_item)
class ReadParquetArrayItemInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 6
        elem_type = args[4].dtype
        # Implement IntegerArrayType(elem_type) when this code path is restored.
        return signature(
            ArrayItemArrayType(_get_series_array_type(elem_type)), *unliteral_all(args)
        )


@infer_global(read_parquet_arrow_array)
class ReadParquetArrowArrayInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 5
        return signature(args[4].instance_type, *unliteral_all(args[:4]))


import llvmlite.binding as ll
from llvmlite import ir as lir
from numba.core import cgutils
from numba.core.imputils import lower_builtin
from numba.np.arrayobj import make_array

from bodo.config import _has_pyarrow

if _has_pyarrow:
    from bodo.io import parquet_cpp

    ll.add_symbol("get_dataset_reader", parquet_cpp.get_dataset_reader)
    ll.add_symbol("del_dataset_reader", parquet_cpp.del_dataset_reader)
    ll.add_symbol("pq_get_size", parquet_cpp.pq_get_size)
    ll.add_symbol("pq_read", parquet_cpp.pq_read)
    ll.add_symbol("pq_read_string", parquet_cpp.pq_read_string)
    ll.add_symbol("pq_read_list_string", parquet_cpp.pq_read_list_string)
    ll.add_symbol("pq_read_array_item", parquet_cpp.pq_read_array_item)
    ll.add_symbol("pq_read_arrow_array", parquet_cpp.pq_read_arrow_array)
    ll.add_symbol("pq_write", parquet_cpp.pq_write)


@lower_builtin(get_column_size_parquet, types.Opaque("arrow_reader"), types.intp)
def pq_size_lower(context, builder, sig, args):
    fnty = lir.FunctionType(
        lir.IntType(64), [lir.IntType(8).as_pointer(), lir.IntType(64)]
    )
    fn = builder.module.get_or_insert_function(fnty, name="pq_get_size")
    return builder.call(fn, args)


@lower_builtin(
    read_parquet,
    types.Opaque("arrow_reader"),
    types.intp,
    types.intp,
    types.Array,
    types.int32,
)
def pq_read_lower(context, builder, sig, args):
    fnty = lir.FunctionType(
        lir.IntType(64),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(64),
            lir.IntType(8).as_pointer(),
            lir.IntType(32),
            lir.IntType(8).as_pointer(),
        ],
    )
    out_array = make_array(sig.args[3])(context, builder, args[3])
    zero_ptr = context.get_constant_null(types.voidptr)

    fn = builder.module.get_or_insert_function(fnty, name="pq_read")
    ret = builder.call(
        fn,
        [
            args[0],
            args[1],
            args[2],
            builder.bitcast(out_array.data, lir.IntType(8).as_pointer()),
            args[4],
            zero_ptr,
        ],
    )
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
    return ret


########################## read nullable int array ###########################


@lower_builtin(
    read_parquet,
    types.Opaque("arrow_reader"),
    types.intp,
    types.intp,
    IntegerArrayType,
    types.int32,
)
@lower_builtin(
    read_parquet,
    types.Opaque("arrow_reader"),
    types.intp,
    types.intp,
    BooleanArrayType,
    types.int32,
)
@lower_builtin(
    read_parquet,
    types.Opaque("arrow_reader"),
    types.intp,
    types.intp,
    DecimalArrayType,
    types.int32,
)
@lower_builtin(
    read_parquet,
    types.Opaque("arrow_reader"),
    types.intp,
    types.intp,
    datetime_date_array_type,
    types.int32,
)
def pq_read_int_arr_lower(context, builder, sig, args):
    fnty = lir.FunctionType(
        lir.IntType(64),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(64),
            lir.IntType(8).as_pointer(),
            lir.IntType(32),
            lir.IntType(8).as_pointer(),
        ],
    )
    int_arr_typ = sig.args[3]
    int_arr = cgutils.create_struct_proxy(int_arr_typ)(context, builder, args[3])
    dtype = int_arr_typ.dtype
    if isinstance(int_arr_typ, DecimalArrayType):
        dtype = bodo.libs.decimal_arr_ext.int128_type
    if int_arr_typ == datetime_date_array_type:
        dtype = types.int64
    data_typ = types.Array(dtype, 1, "C")
    data_array = make_array(data_typ)(context, builder, int_arr.data)
    null_arr_typ = types.Array(types.uint8, 1, "C")
    bitmap = make_array(null_arr_typ)(context, builder, int_arr.null_bitmap)

    fn = builder.module.get_or_insert_function(fnty, name="pq_read")
    ret = builder.call(
        fn,
        [
            args[0],
            args[1],
            args[2],
            builder.bitcast(data_array.data, lir.IntType(8).as_pointer()),
            args[4],
            builder.bitcast(bitmap.data, lir.IntType(8).as_pointer()),
        ],
    )
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
    return ret


############################## read strings ###############################


@lower_builtin(
    read_parquet_str, types.Opaque("arrow_reader"), types.intp, types.intp, types.intp
)
def pq_read_string_lower(context, builder, sig, args):

    string_array = context.make_helper(builder, string_array_type)
    array_item_data_type = ArrayItemArrayType(char_arr_type)
    array_item_array = context.make_helper(builder, array_item_data_type)

    fnty = lir.FunctionType(
        lir.IntType(32),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(64),
            lir.IntType(8).as_pointer().as_pointer(),
        ],
    )

    fn = builder.module.get_or_insert_function(fnty, name="pq_read_string")
    builder.call(
        fn,
        [
            args[0],
            args[1],
            args[2],
            array_item_array._get_ptr_by_name("meminfo"),
        ],
    )
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    string_array.data = array_item_array._getvalue()
    return string_array._getvalue()


############################## read list of strings ###############################


@lower_builtin(
    read_parquet_list_str,
    types.Opaque("arrow_reader"),
    types.intp,
    types.intp,
    types.intp,
)
def pq_read_list_string_lower(context, builder, sig, args):

    # construct array and payload
    typ = sig.return_type
    array_item_array_from_cpp = context.make_helper(builder, typ)

    # read payload data
    fnty = lir.FunctionType(
        lir.IntType(32),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(64),
            lir.IntType(8).as_pointer().as_pointer(),  # meminfo
        ],
    )

    fn = builder.module.get_or_insert_function(fnty, name="pq_read_list_string")
    _res = builder.call(
        fn,
        [
            args[0],
            args[1],
            args[2],
            array_item_array_from_cpp._get_ptr_by_name("meminfo"),
        ],
    )
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    return array_item_array_from_cpp._getvalue()


############################## read list of items ###############################


@lower_builtin(
    read_parquet_array_item,
    types.Opaque("arrow_reader"),
    types.intp,
    types.intp,
    types.intp,
    types.int32,
    types.Any,  # item data type which is ignored in lowering
)
def pq_read_array_item_lower(context, builder, sig, args):

    array_item_type = sig.return_type

    # TODO: refactor array(item) payload handling copied from construct_array_item_array
    # create payload type
    payload_type = ArrayItemArrayPayloadType(array_item_type)
    alloc_type = context.get_value_type(payload_type)
    alloc_size = context.get_abi_sizeof(alloc_type)

    # define dtor
    dtor_fn = define_array_item_dtor(context, builder, array_item_type, payload_type)

    # create meminfo
    meminfo = context.nrt.meminfo_alloc_dtor(
        builder, context.get_constant(types.uintp, alloc_size), dtor_fn
    )
    meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, alloc_type.as_pointer())

    # alloc values in payload
    payload = cgutils.create_struct_proxy(payload_type)(context, builder)
    payload.n_arrays = args[2]  # set size

    # allocate array_info pointers (to be allocated in pq_read_array_item C++ code)
    ll_array_info_type = context.get_value_type(array_info_type)
    data_info_ptr = cgutils.alloca_once(builder, ll_array_info_type)
    offsets_info_ptr = cgutils.alloca_once(builder, ll_array_info_type)
    nulls_info_ptr = cgutils.alloca_once(builder, ll_array_info_type)

    # read payload data
    fnty = lir.FunctionType(
        lir.IntType(32),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(64),
            lir.IntType(32),
            ll_array_info_type.as_pointer(),
            ll_array_info_type.as_pointer(),
            ll_array_info_type.as_pointer(),
        ],
    )

    fn = builder.module.get_or_insert_function(fnty, name="pq_read_array_item")
    builder.call(
        fn,
        [
            args[0],  # dataset reader
            args[1],  # column index
            args[2],  # real column index
            args[4],  # out_dtype (int32 format, see bodo_common.h)
            offsets_info_ptr,  # offsets array info pointer
            data_info_ptr,  # data array info pointer
            nulls_info_ptr,  # null array info pointer
        ],
    )
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    # convert array_info to numpy arrays and set payload attributes
    payload.data = _lower_info_to_array_numpy(
        array_item_type.dtype,
        context,
        builder,
        builder.load(data_info_ptr),
    )
    payload.offsets = _lower_info_to_array_numpy(
        types.Array(offset_type, 1, "C"),
        context,
        builder,
        builder.load(offsets_info_ptr),
    )
    payload.null_bitmap = _lower_info_to_array_numpy(
        types.Array(types.uint8, 1, "C"), context, builder, builder.load(nulls_info_ptr)
    )

    builder.store(payload._getvalue(), meminfo_data_ptr)

    array_item_array = context.make_helper(builder, array_item_type)

    array_item_array.meminfo = meminfo
    ret = array_item_array._getvalue()
    return impl_ret_new_ref(context, builder, array_item_type, ret)


############################ parquet read array table ########################


@lower_builtin(
    read_parquet_arrow_array,
    types.Opaque("arrow_reader"),
    types.intp,
    types.intp,
    types.intp,
)
def pq_read_arrow_array_lower(context, builder, sig, args):

    # TODO attempt more refactoring with code in array.py::info_to_array?

    arr_type = sig.return_type

    def get_num_arrays(arr_typ):
        """ get total number of arrays in nested array """
        if isinstance(arr_typ, ArrayItemArrayType):
            return 1 + get_num_arrays(arr_typ.dtype)
        elif isinstance(arr_typ, StructArrayType):
            return 1 + sum([get_num_arrays(d) for d in arr_typ.data])
        else:
            return 1

    def get_num_infos(arr_typ):
        """get number of array_infos that need to be returned from
        C++ to reconstruct this array"""
        if isinstance(arr_typ, ArrayItemArrayType):
            # 1 buffer for offsets, 1 buffer for nulls + children buffer count
            return 2 + get_num_infos(arr_typ.dtype)
        elif isinstance(arr_typ, StructArrayType):
            # 1 for nulls + children buffer count
            return 1 + sum([get_num_infos(d) for d in arr_typ.data])
        elif arr_typ == string_array_type:
            # C++ will just use one array_info
            return 1
        else:
            # for primitive types: nulls and data
            # NOTE for non-nullable arrays C++ will still return two
            # buffers since it doesn't know that the Arrow array of
            # primitive values is going to be converted to a Numpy array
            # (all Arrow arrays are nullable)
            return 2

    n = get_num_arrays(arr_type)
    # allocate zero-initialized array of lengths for each array in
    # nested datastructure (to be filled out by C++)
    lengths = cgutils.pack_array(
        builder, [lir.Constant(lir.IntType(64), 0) for _ in range(n)]
    )
    lengths_ptr = cgutils.alloca_once_value(builder, lengths)
    # allocate array of null pointers for each buffer in the
    # nested datastructure (to be filled out by C++ as pointers to array_info)
    nullptr = lir.Constant(lir.IntType(8).as_pointer(), None)
    array_infos = cgutils.pack_array(
        builder, [nullptr for _ in range(get_num_infos(arr_type))]
    )
    array_infos_ptr = cgutils.alloca_once_value(builder, array_infos)

    # call C++ info_to_nested_array to fill lengths and array_info arrays
    # each array_info corresponds to one individual buffer (can be
    # offsets, null or data buffer)
    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(8).as_pointer(),  # dataset reader
            lir.IntType(64),  # column index
            lir.IntType(64),  # real column index
            lir.IntType(64),  # number of fields in column
            lir.IntType(64).as_pointer(),  # lengths array
            lir.IntType(8).as_pointer().as_pointer(),  # array of array_info*
        ],
    )
    fn_tp = builder.module.get_or_insert_function(fnty, name="pq_read_arrow_array")
    builder.call(
        fn_tp,
        [
            args[0],  # dataset reader
            args[1],  # column index
            args[2],  # real column index
            args[3],  # number of fields in column
            builder.bitcast(lengths_ptr, lir.IntType(64).as_pointer()),
            builder.bitcast(array_infos_ptr, lir.IntType(8).as_pointer().as_pointer()),
        ],
    )
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    # generate code recursively to construct nested arrays from buffers
    # returned from C++
    arr, _, _ = bodo.libs.array.nested_to_array(
        context, builder, arr_type, lengths_ptr, array_infos_ptr, 0, 0
    )
    return arr


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
        fn_tp = builder.module.get_or_insert_function(fnty, name="pq_write")
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
