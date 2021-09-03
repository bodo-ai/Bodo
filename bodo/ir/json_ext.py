# Copyright (C) 2019 Bodo Inc. All rights reserved.
import numba
import numpy as np  # noqa
import pandas as pd  # noqa
from numba.core import ir, ir_utils, typeinfer, types
from numba.core.ir_utils import compile_to_numba_ir, replace_arg_nodes

import bodo
import bodo.ir.connector
from bodo import objmode  # noqa
from bodo.libs.str_ext import string_type
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.utils.utils import check_java_installation  # noqa
from bodo.utils.utils import sanitize_varname


class JsonReader(ir.Stmt):
    def __init__(
        self,
        df_out,
        loc,
        out_vars,
        out_types,
        file_name,
        df_colnames,
        orient,
        convert_dates,
        precise_float,
        lines,
        compression,
    ):
        self.connector_typ = "json"
        self.df_out = df_out  # used only for printing
        self.loc = loc
        self.out_vars = out_vars
        self.out_types = out_types
        self.file_name = file_name
        self.df_colnames = df_colnames
        self.orient = orient
        self.convert_dates = convert_dates
        self.precise_float = precise_float
        self.lines = lines
        self.compression = compression

    def __repr__(self):  # pragma: no cover
        return "{} = ReadJson(file={}, col_names={}, types={}, vars={})".format(
            self.df_out, self.file_name, self.df_colnames, self.out_types, self.out_vars
        )


import llvmlite.binding as ll

from bodo.io import json_cpp

ll.add_symbol("json_file_chunk_reader", json_cpp.json_file_chunk_reader)

json_file_chunk_reader = types.ExternalFunction(
    "json_file_chunk_reader",
    bodo.ir.connector.stream_reader_type(
        types.voidptr,
        types.bool_,
        types.bool_,
        types.int64,
        types.voidptr,
        types.voidptr,
    ),
)


def remove_dead_json(
    json_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):
    # TODO
    new_df_colnames = []
    new_out_vars = []
    new_out_types = []

    for i, col_var in enumerate(json_node.out_vars):
        if col_var.name in lives:
            new_df_colnames.append(json_node.df_colnames[i])
            new_out_vars.append(json_node.out_vars[i])
            new_out_types.append(json_node.out_types[i])

    json_node.df_colnames = new_df_colnames
    json_node.out_vars = new_out_vars
    json_node.out_types = new_out_types

    if len(json_node.out_vars) == 0:
        return None

    return json_node


def json_distributed_run(
    json_node, array_dists, typemap, calltypes, typingctx, targetctx
):
    parallel = False
    if array_dists is not None:
        parallel = True
        for v in json_node.out_vars:
            if (
                array_dists[v.name] != distributed_pass.Distribution.OneD
                and array_dists[v.name] != distributed_pass.Distribution.OneD_Var
            ):
                parallel = False

    n_cols = len(json_node.out_vars)
    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    # get column variables
    arg_names = ", ".join("arr" + str(i) for i in range(n_cols))
    func_text = "def json_impl(fname):\n"
    func_text += "    ({},) = _json_reader_py(fname)\n".format(arg_names)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    json_impl = loc_vars["json_impl"]
    json_reader_py = _gen_json_reader_py(
        json_node.df_colnames,
        json_node.out_types,
        typingctx,
        targetctx,
        parallel,
        json_node.orient,
        json_node.convert_dates,
        json_node.precise_float,
        json_node.lines,
        json_node.compression,
    )
    f_block = compile_to_numba_ir(
        json_impl,
        {"_json_reader_py": json_reader_py},
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=(string_type,),
        typemap=typemap,
        calltypes=calltypes,
    ).blocks.popitem()[1]
    replace_arg_nodes(f_block, [json_node.file_name])
    nodes = f_block.body[:-3]
    for i in range(len(json_node.out_vars)):
        nodes[-len(json_node.out_vars) + i].target = json_node.out_vars[i]
    return nodes


numba.parfors.array_analysis.array_analysis_extensions[
    JsonReader
] = bodo.ir.connector.connector_array_analysis
distributed_analysis.distributed_analysis_extensions[
    JsonReader
] = bodo.ir.connector.connector_distributed_analysis
typeinfer.typeinfer_extensions[JsonReader] = bodo.ir.connector.connector_typeinfer
# add call to visit json variable
ir_utils.visit_vars_extensions[JsonReader] = bodo.ir.connector.visit_vars_connector
ir_utils.remove_dead_extensions[JsonReader] = remove_dead_json
numba.core.analysis.ir_extension_usedefs[
    JsonReader
] = bodo.ir.connector.connector_usedefs
ir_utils.copy_propagate_extensions[JsonReader] = bodo.ir.connector.get_copies_connector
ir_utils.apply_copy_propagate_extensions[
    JsonReader
] = bodo.ir.connector.apply_copies_connector
ir_utils.build_defs_extensions[
    JsonReader
] = bodo.ir.connector.build_connector_definitions
distributed_pass.distributed_run_extensions[JsonReader] = json_distributed_run

# XXX: temporary fix pending Numba's #3378
# keep the compiled functions around to make sure GC doesn't delete them and
# the reference to the dynamic function inside them
# (numba/lowering.py:self.context.add_dynamic_addr ...)
compiled_funcs = []


def _gen_json_reader_py(
    col_names,
    col_typs,
    typingctx,
    targetctx,
    parallel,
    orient,
    convert_dates,
    precise_float,
    lines,
    compression,
):
    # TODO: support non-numpy types like strings
    sanitized_cnames = [sanitize_varname(c) for c in col_names]
    date_inds = ", ".join(
        str(i) for i, t in enumerate(col_typs) if t.dtype == types.NPDatetime("ns")
    )
    typ_strs = ", ".join(
        [
            "{}='{}'".format(s_cname, bodo.ir.csv_ext._get_dtype_str(t))
            for s_cname, t in zip(sanitized_cnames, col_typs)
        ]
    )
    pd_dtype_strs = ", ".join(
        [
            "'{}':{}".format(cname, bodo.ir.csv_ext._get_pd_dtype_str(t))
            for cname, t in zip(col_names, col_typs)
        ]
    )

    # With Arrow 2.0.0, gzip and bz2 map to gzip and bz2 directly
    # and not GZIP and BZ2 like they used to.
    if compression is None:
        compression = "uncompressed"  # Arrow's representation

    func_text = "def json_reader_py(fname):\n"
    # check_java_installation is a check for hdfs that java is installed
    func_text += "  check_java_installation(fname)\n"
    # if it's an s3 url, get the region and pass it into the c++ code
    func_text += f"  bucket_region = bodo.io.fs_io.get_s3_bucket_region_njit(fname, parallel={parallel})\n"
    func_text += "  f_reader = bodo.ir.json_ext.json_file_chunk_reader(bodo.libs.str_ext.unicode_to_utf8(fname), "
    func_text += "    {}, {}, -1, bodo.libs.str_ext.unicode_to_utf8('{}'), bodo.libs.str_ext.unicode_to_utf8(bucket_region) )\n".format(
        lines, parallel, compression
    )
    # Check if there was an error in the C++ code. If so, raise it.
    func_text += "  bodo.utils.utils.check_and_propagate_cpp_exception()\n"
    func_text += "  if bodo.utils.utils.is_null_pointer(f_reader):\n"
    func_text += "      raise FileNotFoundError('File does not exist')\n"
    func_text += "  with objmode({}):\n".format(typ_strs)
    func_text += "    df = pd.read_json(f_reader, orient='{}',\n".format(orient)
    func_text += "       convert_dates = {}, \n".format(convert_dates)
    func_text += "       precise_float={}, \n".format(precise_float)
    func_text += "       lines={}, \n".format(lines)
    func_text += "       dtype={{{}}},\n".format(pd_dtype_strs)
    func_text += "       )\n"
    for s_cname, cname in zip(sanitized_cnames, col_names):
        func_text += "    if len(df) > 0:\n"
        func_text += "        {} = df['{}'].values\n".format(s_cname, cname)
        func_text += "    else:\n"
        func_text += "        {} = np.array([])\n".format(s_cname)
    func_text += "  return ({},)\n".format(", ".join(sc for sc in sanitized_cnames))

    glbls = globals()  # TODO: fix globals after Numba's #3355 is resolved
    # {'objmode': objmode, 'json_file_chunk_reader': json_file_chunk_reader,
    # 'pd': pd, 'np': np}
    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    json_reader_py = loc_vars["json_reader_py"]

    # TODO: no_cpython_wrapper=True crashes for some reason
    jit_func = numba.njit(json_reader_py)
    compiled_funcs.append(jit_func)
    return jit_func
