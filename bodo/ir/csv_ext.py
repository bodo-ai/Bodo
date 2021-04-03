# Copyright (C) 2019 Bodo Inc. All rights reserved.

import numba
import numpy as np  # noqa
import pandas as pd  # noqa
from numba.core import ir, ir_utils, typeinfer, types
from numba.core.ir_utils import compile_to_numba_ir, replace_arg_nodes

import bodo
import bodo.ir.connector
from bodo import objmode  # noqa
from bodo.hiframes.datetime_date_ext import datetime_date_type
from bodo.hiframes.pd_categorical_ext import (
    CategoricalArrayType,
    PDCategoricalDtype,
)
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import string_type
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.utils.utils import check_java_installation  # noqa
from bodo.utils.utils import sanitize_varname


class CsvReader(ir.Stmt):
    def __init__(
        self,
        file_name,
        df_out,
        sep,
        df_colnames,
        out_vars,
        out_types,
        usecols,
        loc,
        header,
        compression,
        skiprows=0,
    ):
        self.connector_typ = "csv"
        self.file_name = file_name
        self.df_out = df_out  # used only for printing
        self.sep = sep
        self.df_colnames = df_colnames
        self.out_vars = out_vars
        self.out_types = out_types
        self.usecols = usecols
        self.loc = loc
        self.skiprows = skiprows
        self.header = header
        self.compression = compression

    def __repr__(self):  # pragma: no cover
        return "{} = ReadCsv(file={}, col_names={}, types={}, vars={})".format(
            self.df_out, self.file_name, self.df_colnames, self.out_types, self.out_vars
        )


import llvmlite.binding as ll

from bodo.io import csv_cpp

ll.add_symbol("csv_file_chunk_reader", csv_cpp.csv_file_chunk_reader)

csv_file_chunk_reader = types.ExternalFunction(
    "csv_file_chunk_reader",
    bodo.ir.connector.stream_reader_type(
        types.voidptr,
        types.bool_,
        types.int64,
        types.int64,
        types.bool_,
        types.voidptr,
        types.voidptr,
    ),
)


def remove_dead_csv(
    csv_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):
    # TODO
    new_df_colnames = []
    new_out_vars = []
    new_out_types = []
    new_usecols = []

    for i, col_var in enumerate(csv_node.out_vars):
        if col_var.name in lives:
            new_df_colnames.append(csv_node.df_colnames[i])
            new_out_vars.append(csv_node.out_vars[i])
            new_out_types.append(csv_node.out_types[i])
            new_usecols.append(csv_node.usecols[i])

    csv_node.df_colnames = new_df_colnames
    csv_node.out_vars = new_out_vars
    csv_node.out_types = new_out_types
    csv_node.usecols = new_usecols

    if len(csv_node.out_vars) == 0:
        return None

    return csv_node


def csv_distributed_run(
    csv_node, array_dists, typemap, calltypes, typingctx, targetctx
):
    parallel = False
    if array_dists is not None:
        parallel = True
        for v in csv_node.out_vars:
            if (
                array_dists[v.name] != distributed_pass.Distribution.OneD
                and array_dists[v.name] != distributed_pass.Distribution.OneD_Var
            ):
                parallel = False

    n_cols = len(csv_node.out_vars)
    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    # get column variables
    arg_names = ", ".join("arr" + str(i) for i in range(n_cols))
    func_text = "def csv_impl(fname):\n"
    func_text += "    ({},) = _csv_reader_py(fname)\n".format(arg_names)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    csv_impl = loc_vars["csv_impl"]

    csv_reader_py = _gen_csv_reader_py(
        csv_node.df_colnames,
        csv_node.out_types,
        csv_node.usecols,
        csv_node.sep,
        typingctx,
        targetctx,
        parallel,
        csv_node.skiprows,
        csv_node.header,
        csv_node.compression,
    )

    f_block = compile_to_numba_ir(
        csv_impl,
        {"_csv_reader_py": csv_reader_py},
        typingctx,
        (string_type,),
        typemap,
        calltypes,
    ).blocks.popitem()[1]
    replace_arg_nodes(f_block, [csv_node.file_name])
    nodes = f_block.body[:-3]
    for i in range(len(csv_node.out_vars)):
        nodes[-len(csv_node.out_vars) + i].target = csv_node.out_vars[i]

    return nodes


numba.parfors.array_analysis.array_analysis_extensions[
    CsvReader
] = bodo.ir.connector.connector_array_analysis
distributed_analysis.distributed_analysis_extensions[
    CsvReader
] = bodo.ir.connector.connector_distributed_analysis
typeinfer.typeinfer_extensions[CsvReader] = bodo.ir.connector.connector_typeinfer
ir_utils.visit_vars_extensions[CsvReader] = bodo.ir.connector.visit_vars_connector
ir_utils.remove_dead_extensions[CsvReader] = remove_dead_csv
numba.core.analysis.ir_extension_usedefs[
    CsvReader
] = bodo.ir.connector.connector_usedefs
ir_utils.copy_propagate_extensions[CsvReader] = bodo.ir.connector.get_copies_connector
ir_utils.apply_copy_propagate_extensions[
    CsvReader
] = bodo.ir.connector.apply_copies_connector
ir_utils.build_defs_extensions[
    CsvReader
] = bodo.ir.connector.build_connector_definitions
distributed_pass.distributed_run_extensions[CsvReader] = csv_distributed_run


def _get_dtype_str(t):
    dtype = t.dtype
    if isinstance(dtype, PDCategoricalDtype):
        cat_arr = CategoricalArrayType(dtype)
        # HACK: add cat type to numba.core.types
        # FIXME: fix after Numba #3372 is resolved
        cat_arr_name = "CategoricalArrayType" + str(ir_utils.next_label())
        setattr(types, cat_arr_name, cat_arr)
        return cat_arr_name

    if dtype == types.NPDatetime("ns"):
        dtype = 'NPDatetime("ns")'

    if t == string_array_type:
        # HACK: add string_array_type to numba.core.types
        # FIXME: fix after Numba #3372 is resolved
        types.string_array_type = string_array_type
        return "string_array_type"

    if isinstance(t, IntegerArrayType):
        # HACK: same issue as above
        t_name = "int_arr_{}".format(dtype)
        setattr(types, t_name, t)
        return t_name

    if t == boolean_array:
        types.boolean_array = boolean_array
        return "boolean_array"

    if dtype == types.bool_:
        dtype = "bool_"

    if dtype == datetime_date_type:
        return "datetime_date_array_type"

    return "{}[::1]".format(dtype)


def _get_pd_dtype_str(t):
    dtype = t.dtype

    if isinstance(dtype, PDCategoricalDtype):
        return "pd.CategoricalDtype({})".format(dtype.categories)

    if dtype == types.NPDatetime("ns"):
        return "str"

    if t == string_array_type:
        return "str"

    # nullable int array
    if isinstance(t, IntegerArrayType):
        return '"{}Int{}"'.format("" if dtype.signed else "U", dtype.bitwidth)

    if t == boolean_array:
        return "np.bool_"

    return "np.{}".format(dtype)


# XXX: temporary fix pending Numba's #3378
# keep the compiled functions around to make sure GC doesn't delete them and
# the reference to the dynamic function inside them
# (numba/lowering.py:self.context.add_dynamic_addr ...)
compiled_funcs = []


def _gen_csv_reader_py(
    col_names,
    col_typs,
    usecols,
    sep,
    typingctx,
    targetctx,
    parallel,
    skiprows,
    header,
    compression,
):
    # TODO: support non-numpy types like strings
    sanitized_cnames = [sanitize_varname(c) for c in col_names]
    date_inds = ", ".join(
        str(i) for i, t in enumerate(col_typs) if t.dtype == types.NPDatetime("ns")
    )
    typ_strs = ", ".join(
        [
            "{}='{}'".format(s_cname, _get_dtype_str(t))
            for s_cname, t in zip(sanitized_cnames, col_typs)
        ]
    )

    # Pandas' `read_csv` and Bodo's `read_csv` are not exactly equivalent,
    # for instance in a column of `int64` if there is a missing entry,
    # pandas would convert it to a `float64` column whereas Bodo would use a
    # `Int64` type (nullable integers), etc. We discovered a performance bug with
    # certain nullable types, notably `Int64`, in read_csv, i.e. when we
    # specify the `Int64` dtype in `pd.read_csv`, the performance is very poor.
    # Interestingly, if we do `pd.read_csv` without `dtype` argument and then
    # simply do `df.astype` right after, we do not face the performance
    # penalty. However, when reading strings, if we have missing entries,
    # doing `df.astype` would convert those entries to literally the
    # string values "nan". This is not desirable. Ideally we would use the
    # nullable string type ("string") which would not have this issue, but
    # unfortunately the performance is slow (in both `pd.read_csv` and `df.astype`).
    # Therefore, we have the workaround below where we specify the `dtype` for strings
    # (`str`) directly in `pd.read_csv` (there's no performance penalty, we checked),
    # and specify the rest of the dtypes in the `df.astype` call.

    # dtypes to specify directly in the `pd.read_csv` call
    pd_read_csv_dtype_strs = ", ".join(
        [
            "{}:{}".format(idx, _get_pd_dtype_str(t))
            for idx, t in zip(usecols, col_typs)
            if _get_pd_dtype_str(t) == "str"
        ]
    )
    # dtypes to specify in the `df.astype` call done right after the `pd.read_csv` call
    df_astype_dtype_strs = ", ".join(
        [
            "{}:{}".format(idx, _get_pd_dtype_str(t))
            for idx, t in zip(usecols, col_typs)
            if _get_pd_dtype_str(t) != "str"
        ]
    )
    # here, header can either be:
    #  0 meaning the first row of the file(s) is the header row
    #  None meaning the file(s) does not contain header
    has_header = header == 0

    # With Arrow 2.0.0, gzip and bz2 map to gzip and bz2 directly
    # and not GZIP and BZ2 like they used to.
    if compression is None:
        compression = "uncompressed"  # Arrow's representation

    func_text = "def csv_reader_py(fname):\n"
    # check_java_installation is a check for hdfs that java is installed
    func_text += "  check_java_installation(fname)\n"
    func_text += "  skiprows = {}\n".format(skiprows)
    # if it's an s3 url, get the region and pass it into the c++ code
    func_text += "  bucket_region = bodo.io.fs_io.get_s3_bucket_region_njit(fname)\n"
    func_text += "  f_reader = bodo.ir.csv_ext.csv_file_chunk_reader(bodo.libs.str_ext.unicode_to_utf8(fname), "
    func_text += "    {}, skiprows, -1, {}, bodo.libs.str_ext.unicode_to_utf8('{}'), bodo.libs.str_ext.unicode_to_utf8(bucket_region) )\n".format(
        parallel, has_header, compression
    )
    # Check if there was an error in the C++ code. If so, raise it.
    func_text += "  bodo.utils.utils.check_and_propagate_cpp_exception()\n"
    func_text += "  if bodo.utils.utils.is_null_pointer(f_reader):\n"
    func_text += "      raise FileNotFoundError('File does not exist')\n"
    func_text += "  with objmode({}):\n".format(typ_strs)
    func_text += "    df = pd.read_csv(f_reader,\n"
    # header is always None here because header information was found in untyped pass.
    # this pd.read_csv() happens at runtime and is passing a file reader(f_reader)
    # to pandas. f_reader skips the header, so we have to tell pandas header=None.
    func_text += "       header=None,\n"
    func_text += "       parse_dates=[{}],\n".format(date_inds)
    # Check explanation near the declaration of `pd_read_csv_dtype_strs` for why we specify
    # only some types here directly
    func_text += "       dtype={{{}}},\n".format(pd_read_csv_dtype_strs)
    func_text += "       usecols={}, sep='{}', low_memory=False)\n".format(usecols, sep)
    # Check explanation near the declaration of `df_astype_dtype_strs` for why we specify
    # some types here rather than directly in the `pd.read_csv` call.
    func_text += "    df = df.astype({{{}}}, copy=False)\n".format(df_astype_dtype_strs)
    for col_idx, s_cname in zip(usecols, sanitized_cnames):
        func_text += "    {} = df[{}].values\n".format(s_cname, col_idx)
    func_text += "  return ({},)\n".format(", ".join(sc for sc in sanitized_cnames))
    glbls = globals()  # TODO: fix globals after Numba's #3355 is resolved
    # {'objmode': objmode, 'csv_file_chunk_reader': csv_file_chunk_reader,
    # 'pd': pd, 'np': np}
    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    csv_reader_py = loc_vars["csv_reader_py"]

    # TODO: no_cpython_wrapper=True crashes for some reason
    jit_func = numba.njit(csv_reader_py)
    compiled_funcs.append(jit_func)
    return jit_func
