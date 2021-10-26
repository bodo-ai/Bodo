# Copyright (C) 2019 Bodo Inc. All rights reserved.
import os
import warnings
from collections import defaultdict

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
from numba.core import ir, types
from numba.core.ir_utils import (
    compile_to_numba_ir,
    get_definition,
    guard,
    mk_unique_var,
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
from bodo.io.fs_io import get_hdfs_fs, get_s3_fs_from_path
from bodo.libs.array import (
    delete_table,
    info_from_table,
    info_to_array,
    table_type,
)
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.binary_arr_ext import binary_array_type, bytes_type
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.decimal_arr_ext import DecimalArrayType
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

    def __init__(self, columns, storage_options=None):
        self.columns = columns  # columns to select from parquet dataset
        self.storage_options = storage_options

    def get_schema(self, fname):
        try:
            return parquet_file_schema(
                fname,
                selected_columns=self.columns,
                storage_options=self.storage_options,
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

    def gen_parquet_read(self, file_name, lhs, columns, storage_options=None):
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
                file_info=ParquetFileInfo(columns, storage_options=storage_options),
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
                    ) = typ.schema
                    got_schema = True
            if not got_schema:
                (
                    col_names,
                    col_types,
                    index_col,
                    col_indices,
                    partition_names,
                ) = parquet_file_schema(
                    file_name_str, columns, storage_options=storage_options
                )
        else:
            col_names_total = list(table_types.keys())
            col_types_total = [t for t in table_types.values()]
            index_col = "index" if "index" in col_names_total else None
            # TODO: allow specifying types of only selected columns
            if columns is None:
                selected_columns = col_names_total
            else:
                selected_columns = columns
            col_indices = [col_names_total.index(c) for c in selected_columns]
            col_types = [
                col_types_total[col_names_total.index(c)] for c in selected_columns
            ]
            col_names = selected_columns
            index_col = index_col if index_col in col_names else None
            partition_names = []

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
                col_types,
                data_arrs,
                loc,
                partition_names,
                storage_options,
            )
        ]

        return col_names, data_arrs, index_col, nodes


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
    filter_str = "None"
    filter_map, filter_vars = bodo.ir.connector.generate_filter_map(pq_node.filters)
    extra_args = ", ".join(filter_map.values())
    if filter_map:
        filter_str = "[{}]".format(
            ", ".join(
                "[{}]".format(
                    ", ".join(
                        f"('{v[0]}', '{v[1]}', {filter_map[v[2].name]})"
                        for v in predicate
                    )
                )
                for predicate in pq_node.filters
            )
        )
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
        pq_node.out_types,
        pq_node.storage_options,
        pq_node.partition_names,
        filter_str,
        extra_args,
        typingctx,
        targetctx,
        parallel,
        meta_head_only_info,
    )
    arg_types = (string_type,) + tuple(typemap[v.name] for v in filter_vars)
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

    for i in range(n_cols):
        nodes[i - n_cols].target = pq_node.out_vars[i]

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
    exec(func_text, globals(), loc_vars)
    return loc_vars["impl"]


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
    out_types,
    storage_options,
    partition_names,
    filter_str,
    extra_args,
    typingctx,
    targetctx,
    parallel,
    meta_head_only_info,
):
    if len(parallel) > 0:
        # in parallel read, we assume all columns are parallel
        assert col_names == parallel
    is_parallel = len(parallel) > 0
    comma = "," if extra_args else ""
    func_text = f"def pq_reader_py(fname,{extra_args}):\n"
    # if it's an s3 url, get the region and pass it into the c++ code
    func_text += f"    ev = bodo.utils.tracing.Event('read_parquet', {is_parallel})\n"
    func_text += "    ev.add_attribute('fname', fname)\n"
    func_text += f"    bucket_region = bodo.io.fs_io.get_s3_bucket_region_njit(fname, parallel={is_parallel})\n"
    func_text += (
        f'    filters = get_filters_pyobject("{filter_str}", ({extra_args}{comma}))\n'
    )

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
    local_types = {}
    for col_in_idx, c_typ in zip(col_indices, out_types):
        local_types[f"col_{col_in_idx}_type"] = c_typ

    # Get list of selected columns to pass to C++ (not including partition
    # columns, since they are not in the parquet files).
    # C++ doesn't need to know the order of output columns, and to simplify
    # the code we will pass the indices of columns in the parquet file sorted.
    # C++ code will add partition columns to the end of its output table.
    selected_cols = sorted(
        [
            col_in_idx
            for c_name, col_in_idx in zip(sanitized_col_names, col_indices)
            if c_name not in partition_names
        ]
    )

    # Tell C++ which columns in the parquet file are nullable, since there
    # are some types like integer which Arrow always considers to be nullable
    # but pandas might not. This is mainly intended to tell C++ which Int/Bool
    # arrays require null bitmap and which don't

    def is_nullable(typ):
        return bodo.utils.utils.is_array_typ(typ, False) and (
            not isinstance(typ, types.Array)  # or is_dtype_nullable(typ.dtype)
        )

    nullable_cols = [
        int(is_nullable(out_types[col_indices.index(col_in_idx)]))
        for col_in_idx in selected_cols
    ]

    # partition_names is the list of *all* partition column names in the
    # parquet dataset as given by pyarrow.parquet.ParquetDataset.
    # We pass selected partition columns to C++, in the order and index used
    # by pyarrow.parquet.ParquetDataset (e.g. 0 is the first partition col)
    # We also pass the dtype of categorical codes
    sel_partition_names = []
    selected_partition_cols = []
    partition_col_cat_dtypes = []
    for i, part_name in enumerate(partition_names):
        try:
            col_out_idx = sanitized_col_names.index(part_name)
        except ValueError:
            # this partition column has not been selected for read
            continue
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
    if len(selected_partition_cols) > 0:
        func_text += f"    out_table = pq_read(unicode_to_utf8(fname), {is_parallel}, unicode_to_utf8(bucket_region), filters, storage_options_py, {tot_rows_to_read}, np.array({selected_cols}, dtype=np.int32).ctypes, {len(selected_cols)}, np.array({nullable_cols}, dtype=np.int32).ctypes, np.array({selected_partition_cols}, dtype=np.int32).ctypes, np.array({partition_col_cat_dtypes}, dtype=np.int32).ctypes, {len(selected_partition_cols)}, total_rows_np.ctypes)\n"
    else:
        func_text += f"    out_table = pq_read(unicode_to_utf8(fname), {is_parallel}, unicode_to_utf8(bucket_region), filters, storage_options_py, {tot_rows_to_read}, np.array({selected_cols}, dtype=np.int32).ctypes, {len(selected_cols)}, np.array({nullable_cols}, dtype=np.int32).ctypes, 0, 0, 0, total_rows_np.ctypes)\n"
    func_text += "    check_and_propagate_cpp_exception()\n"

    for i, col_in_idx in enumerate(selected_cols):
        c_name = sanitized_col_names[col_indices.index(col_in_idx)]
        func_text += f"    {c_name} = info_to_array(info_from_table(out_table, {i}), col_{col_in_idx}_type)\n"

    for i, c_name in enumerate(sel_partition_names):
        col_in_idx = col_indices[sanitized_col_names.index(c_name)]
        func_text += f"    {c_name} = info_to_array(info_from_table(out_table, {i + len(selected_cols)}), col_{col_in_idx}_type)\n"

    func_text += "    delete_table(out_table)\n"
    func_text += f"    total_rows = total_rows_np[0]\n"
    func_text += f"    ev.finalize()\n"
    func_text += "    return (total_rows, {},)\n".format(", ".join(sanitized_col_names))
    loc_vars = {}
    glbs = {
        "info_to_array": info_to_array,
        "info_from_table": info_from_table,
        "delete_table": delete_table,
        "check_and_propagate_cpp_exception": check_and_propagate_cpp_exception,
        "pq_read": _pq_read,
        "unicode_to_utf8": unicode_to_utf8,
        "get_filters_pyobject": get_filters_pyobject,
        "get_storage_options_pyobject": get_storage_options_pyobject,
        "np": np,
        "pd": pd,
        "bodo": bodo,
    }
    glbs.update(local_types)

    exec(func_text, glbs, loc_vars)
    pq_reader_py = loc_vars["pq_reader_py"]

    jit_func = numba.njit(pq_reader_py, no_cpython_wrapper=True)
    return jit_func


def _get_numba_typ_from_pa_typ(pa_typ, is_index, nullable_from_metadata, category_info):
    """return Bodo array type from pyarrow Field (column type)"""
    import pyarrow as pa

    if isinstance(pa_typ.type, pa.ListType):
        return ArrayItemArrayType(
            # nullable_from_metadata is only used for non-nested Int arrays
            _get_numba_typ_from_pa_typ(
                pa_typ.type.value_field, is_index, nullable_from_metadata, category_info
            )
        )

    if isinstance(pa_typ.type, pa.StructType):
        child_types = []
        field_names = []
        for field in pa_typ.flatten():
            field_names.append(field.name.split(".")[-1])
            child_types.append(
                _get_numba_typ_from_pa_typ(
                    field, is_index, nullable_from_metadata, category_info
                )
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

    # Categorical data type
    if isinstance(pa_typ.type, pa.DictionaryType):
        # NOTE: non-string categories seems not possible as of Arrow 4.0
        if pa_typ.type.value_type != pa.string():  # pragma: no cover
            raise BodoError(
                f"Parquet Categorical data type should be string, not {pa_typ.type.value_type}"
            )
        # data type for storing codes
        int_type = _typ_map[pa_typ.type.index_type]
        cat_dtype = PDCategoricalDtype(
            category_info[pa_typ.name],
            bodo.string_type,
            pa_typ.type.ordered,
            int_type=int_type,
        )
        return CategoricalArrayType(cat_dtype)

    if pa_typ.type not in _typ_map:
        raise BodoError("Arrow data type {} not supported yet".format(pa_typ.type))
    dtype = _typ_map[pa_typ.type]

    if dtype == datetime_date_type:
        return datetime_date_array_type

    if dtype == bytes_type:
        return binary_array_type

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


def get_parquet_dataset(
    fpath,
    get_row_counts=True,
    filters=None,
    storage_options=None,
    read_categories=False,
    is_parallel=False,  # only used with get_row_counts=True
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

    import pyarrow as pa
    import pyarrow.parquet as pq
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    fpath = fpath.rstrip("/")
    if fpath.startswith("gs://"):
        fpath = fpath[:1] + "c" + fpath[1:]

    if fpath.startswith("gcs://"):
        try:
            import gcsfs
        except ImportError:
            message = (
                "Couldn't import gcsfs, which is required for Google cloud access."
                " gcsfs can be installed by calling"
                " 'conda install -c conda-forge gcsfs'.\n"
            )
            raise BodoError(message)

    if fpath.startswith("http://"):
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
        if len(fs) == 1:
            return fs[0]
        if fpath.startswith("s3://"):
            fs.append(
                get_s3_fs_from_path(
                    fpath, parallel=parallel, storage_options=storage_options
                )
            )
        elif fpath.startswith("gcs://"):
            # TODO pass storage_options to GCSFileSystem
            google_fs = gcsfs.GCSFileSystem(token=None)
            fs.append(google_fs)
        elif fpath.startswith("http://"):
            fs.append(fsspec.filesystem("http"))
        elif (
            fpath.startswith("hdfs://")
            or fpath.startswith("abfs://")
            or fpath.startswith("abfss://")
        ):  # pragma: no cover
            fs.append(get_hdfs_fs(fpath))
        else:
            fs.append(None)
        return fs[0]

    validate_schema = bodo.parquet_validate_schema
    if get_row_counts or validate_schema:
        # Getting row counts and schema validation is going to be
        # distributed across ranks, so every rank will need a filesystem
        # object to query the metadata of their assigned pieces.
        # There are issues with broadcasting the s3 filesystem object
        # as part of the ParquetDataset, so instead we initialize
        # the filesystem before the broadcast. One of the issues is that
        # our PyArrowS3FS is not correctly pickled (instead what is
        # sent is the underlying pyarrow S3FileSystem). The other issue
        # is that for some reason unpickling seems like it can cause
        # incorrect credential handling state in Arrow or AWS client.
        _ = getfs(parallel=True)

    if bodo.get_rank() == 0:
        nthreads = 1  # number of threads to use on rank 0 to collect metadata
        cpu_count = os.cpu_count()
        if cpu_count is not None and cpu_count > 1:
            nthreads = cpu_count // 2
        try:
            if get_row_counts:
                ev_pq_ds = tracing.Event("pq.ParquetDataset", is_parallel=False)
            pa_default_io_thread_count = pa.io_thread_count()
            pa.set_io_thread_count(nthreads)
            dataset = pq.ParquetDataset(
                fpath,
                filesystem=getfs(),
                filters=filters,
                use_legacy_dataset=True,  # To ensure that ParquetDataset and not ParquetDatasetV2 is used
                validate_schema=False,  # we do validation below if needed
                metadata_nthreads=nthreads,
            )
            # restore pyarrow default IO thread count
            pa.set_io_thread_count(pa_default_io_thread_count)
            if get_row_counts:
                ev_pq_ds.finalize()
            dataset_schema = bodo.io.pa_parquet.get_dataset_schema(dataset)
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
    if get_row_counts:
        ev_bcast.finalize()

    dataset._bodo_total_rows = 0
    if get_row_counts or validate_schema:
        # getting row counts and validating schema requires reading
        # the file metadata from the parquet files and is very expensive
        # for datasets consisting of many files, so we do this in parallel
        if get_row_counts:
            ev_row_counts = tracing.Event("get_row_counts")
        num_pieces = len(dataset.pieces)
        start = get_start(num_pieces, bodo.get_size(), bodo.get_rank())
        end = get_end(num_pieces, bodo.get_size(), bodo.get_rank())
        total_rows_chunk = 0
        total_row_groups_chunk = 0
        piece_nrows_chunk = []
        valid = True  # True if schema of all parquet files match
        dataset._metadata.fs = getfs()
        for p in dataset.pieces[start:end]:
            file_metadata = p.get_metadata()
            if get_row_counts:
                piece_nrows_chunk.append(file_metadata.num_rows)
                total_rows_chunk += file_metadata.num_rows
                total_row_groups_chunk += file_metadata.num_row_groups
            if validate_schema:
                file_schema = file_metadata.schema.to_arrow_schema()
                if dataset_schema != file_schema:  # pragma: no cover
                    # this is the same error message that pyarrow shows
                    print(
                        "Schema in {!s} was different. \n{!s}\n\nvs\n\n{!s}".format(
                            p, file_schema, dataset_schema
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
            if (
                is_parallel
                and bodo.get_rank() == 0
                and total_num_row_groups < bodo.get_size()
            ):
                warnings.warn(
                    BodoWarning(
                        f"""Total number of row groups in parquet dataset {fpath} ({total_num_row_groups}) is too small for effective IO parallelization.
For best performance the number of row groups should be greater than the number of workers ({bodo.get_size()})
"""
                    )
                )
            rows_by_ranks = comm.allgather(piece_nrows_chunk)
            for i, num_rows in enumerate(
                [n for sublist in rows_by_ranks for n in sublist]
            ):
                dataset.pieces[i]._bodo_num_rows = num_rows
            ev_row_counts.add_attribute("total_num_row_groups", total_num_row_groups)
            ev_row_counts.finalize()

    # When we pass in a path instead of a filename or list of files to
    # pq.ParquetDataset, the path of the pieces of the resulting dataset
    # object may not contain the prefix such as hdfs://localhost:9000,
    # so we need to store the information in some way since the
    # code that uses this dataset object needs to know this information.
    # Therefore, in case this prefix is not present, we add it
    # to the dataset object as a new attribute (_prefix).

    # Parse the URI
    fpath_options = urlparse(fpath)
    # Only hdfs seems to be affected based on:
    # https://github.com/apache/arrow/blob/ab307365198d5724a0b3331a05704d0fe4bcd710/python/pyarrow/parquet.py#L50
    # which is called in pq.ParquetDataset.__init__
    # https://github.com/apache/arrow/blob/ab307365198d5724a0b3331a05704d0fe4bcd710/python/pyarrow/parquet.py#L1235
    dataset._prefix = ""
    if fpath_options.scheme in ["hdfs"]:
        # Compute the prefix that is expected to be there in the
        # path of the pieces
        prefix = f"{fpath_options.scheme}://{fpath_options.netloc}"
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


def parquet_file_schema(file_name, selected_columns, storage_options=None):
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

    # NOTE: use arrow schema instead of the dataset schema to avoid issues with
    # names of list types columns (arrow 0.17.0)
    # col_names is an array that contains all the column's name and
    # index's name if there is one, otherwise "__index__level_0"
    # If there is no index at all, col_names will not include anything.
    col_names = pa_schema.names
    index_col, nullable_from_metadata = get_pandas_metadata(pa_schema, num_pieces)
    col_types_total = [
        _get_numba_typ_from_pa_typ(
            pa_schema.field(c),
            c == index_col,
            nullable_from_metadata[c],
            pq_dataset._category_info,
        )
        for c in col_names
    ]

    # add partition column data types if any
    if partition_names:
        col_names += partition_names
        col_types_total += [
            _get_partition_cat_dtype(pq_dataset.partitions.levels[i])
            for i in range(len(partition_names))
        ]

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

    col_indices = [col_names.index(c) for c in selected_columns]
    col_types = [col_types_total[col_names.index(c)] for c in selected_columns]
    col_names = selected_columns
    # TODO: close file?
    return (
        col_names,
        col_types,
        index_col,
        col_indices,
        partition_names,
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
        types.voidptr,
        types.boolean,
        types.voidptr,
        parquet_predicate_type,
        storage_options_dict_type,
        types.int64,
        types.voidptr,
        types.int32,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.int32,
        types.voidptr,
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
