# Copyright (C) 2022 Bodo Inc. All rights reserved.
import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, ir, ir_utils, typeinfer, types
from numba.core.ir_utils import compile_to_numba_ir, replace_arg_nodes
from numba.extending import intrinsic

import bodo
import bodo.ir.connector
import bodo.user_logging
from bodo.io import json_cpp
from bodo.io.fs_io import (
    get_storage_options_pyobject,
    storage_options_dict_type,
)
from bodo.ir.connector import Connector
from bodo.libs.str_ext import string_type
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.utils.utils import (
    check_java_installation,
    sanitize_varname,
)

ll.add_symbol("json_file_chunk_reader", json_cpp.json_file_chunk_reader)


class JsonReader(Connector):
    connector_typ = "json"

    def __init__(
        self,
        df_out_varname: str,
        loc: ir.Loc,
        out_vars: list[ir.Var],
        out_table_col_types,
        file_name,
        out_table_col_names: list[str],
        orient,
        convert_dates,
        precise_float,
        lines,
        compression,
        storage_options,
    ):
        self.df_out_varname = df_out_varname  # used only for printing
        self.loc = loc
        # Each column is returned to a separate variable
        # unlike the C++ readers that return a single TableType
        self.out_vars = out_vars
        self.out_table_col_types = out_table_col_types
        self.file_name = file_name
        self.out_table_col_names = out_table_col_names
        self.orient = orient
        self.convert_dates = convert_dates
        self.precise_float = precise_float
        self.lines = lines
        self.compression = compression
        self.storage_options = storage_options

    def __repr__(self):  # pragma: no cover
        return "{} = ReadJson(file={}, col_names={}, types={}, vars={})".format(
            self.df_out_varname,
            self.file_name,
            self.out_table_col_names,
            self.out_table_col_types,
            self.out_vars,
        )

    def out_vars_and_types(self) -> list[tuple[str, types.Type]]:
        return list(zip((x.name for x in self.out_vars), self.out_table_col_types))


@intrinsic
def json_file_chunk_reader(
    typingctx,
    fname_t,
    lines_t,
    is_parallel_t,
    nrows_t,
    compression_t,
    bucket_region_t,
    storage_options_t,
):
    """
    Interface to json_file_chunk_reader function in C++ library for creating
    the json file reader.
    """
    # TODO: Update storage options to pyobject once the type is updated to do refcounting
    # properly.
    assert (
        storage_options_t == storage_options_dict_type
    ), "Storage options don't match expected type"

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="json_file_chunk_reader"
        )
        obj = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        # json_file_chunk_reader returns a pyobject. We need to wrap the result in the
        # proper return type and create a meminfo.
        ret = cgutils.create_struct_proxy(types.stream_reader_type)(context, builder)
        pyapi = context.get_python_api(builder)
        # borrows and manages a reference for obj (see comments in py_objs.py)
        ret.meminfo = pyapi.nrt_meminfo_new_from_pyobject(
            context.get_constant_null(types.voidptr), obj
        )
        ret.pyobj = obj
        # `nrt_meminfo_new_from_pyobject` increfs the object (holds a reference)
        # so need to decref since the object is not live anywhere else.
        pyapi.decref(obj)
        return ret._getvalue()

    return (
        types.stream_reader_type(
            types.voidptr,  # filename
            types.bool_,  # lines
            types.bool_,  # is_parallel
            types.int64,  # nrows
            types.voidptr,  # compression
            types.voidptr,  # bucket_region
            storage_options_dict_type,  # storage_options dictionary
        ),
        codegen,
    )


def remove_dead_json(
    json_node: JsonReader,
    lives_no_aliases,
    lives,
    arg_aliases,
    alias_map,
    func_ir,
    typemap,
):
    # TODO
    new_col_names = []
    new_out_vars = []
    new_out_types = []

    for i, col_var in enumerate(json_node.out_vars):
        if col_var.name in lives:
            new_col_names.append(json_node.out_table_col_names[i])
            new_out_vars.append(json_node.out_vars[i])
            new_out_types.append(json_node.out_table_col_types[i])

    json_node.out_table_col_names = new_col_names
    json_node.out_vars = new_out_vars
    json_node.out_table_col_types = new_out_types

    if len(json_node.out_vars) == 0:
        return None

    return json_node


def json_distributed_run(
    json_node: JsonReader, array_dists, typemap, calltypes, typingctx, targetctx
):
    # Add debug info about column pruning
    if bodo.user_logging.get_verbose_level() >= 1:
        msg = "Finish column pruning on read_json node:\n%s\nColumns loaded %s\n"
        json_source = json_node.loc.strformat()
        json_cols = json_node.out_table_col_names
        bodo.user_logging.log_message(
            "Column Pruning",
            msg,
            json_source,
            json_cols,
        )
        # Log if any columns use dictionary encoded arrays.
        dict_encoded_cols = [
            c
            for i, c in enumerate(json_node.out_table_col_names)
            if isinstance(
                json_node.out_table_col_types[i],
                bodo.libs.dict_arr_ext.DictionaryArrayType,
            )
        ]
        # TODO: Test. Dictionary encoding isn't supported yet.
        if dict_encoded_cols:
            encoding_msg = "Finished optimized encoding on read_json node:\n%s\nColumns %s using dictionary encoding to reduce memory usage.\n"
            bodo.user_logging.log_message(
                "Dictionary Encoding",
                encoding_msg,
                json_source,
                dict_encoded_cols,
            )

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
        json_node.out_table_col_names,
        json_node.out_table_col_types,
        typingctx,
        targetctx,
        parallel,
        json_node.orient,
        json_node.convert_dates,
        json_node.precise_float,
        json_node.lines,
        json_node.compression,
        json_node.storage_options,
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
    storage_options,
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
    # NOTE: assigning a new variable to make globals used inside objmode local to the
    # function, which avoids objmode caching errors
    func_text += "  df_typeref_2 = df_typeref\n"
    # check_java_installation is a check for hdfs that java is installed
    func_text += "  check_java_installation(fname)\n"
    # if it's an s3 url, get the region and pass it into the c++ code
    func_text += f"  bucket_region = bodo.io.fs_io.get_s3_bucket_region_njit(fname, parallel={parallel})\n"
    # Add a dummy variable to the dict (empty dicts are not yet supported in numba).
    if storage_options is None:
        storage_options = {}
    storage_options["bodo_dummy"] = "dummy"
    func_text += (
        f"  storage_options_py = get_storage_options_pyobject({str(storage_options)})\n"
    )
    func_text += "  f_reader = bodo.ir.json_ext.json_file_chunk_reader(bodo.libs.str_ext.unicode_to_utf8(fname), "
    func_text += "    {}, {}, -1, bodo.libs.str_ext.unicode_to_utf8('{}'), bodo.libs.str_ext.unicode_to_utf8(bucket_region), storage_options_py )\n".format(
        lines, parallel, compression
    )
    func_text += "  if bodo.utils.utils.is_null_pointer(f_reader._pyobj):\n"
    func_text += "      raise FileNotFoundError('File does not exist')\n"
    func_text += f"  with bodo.no_warning_objmode({typ_strs}):\n"
    func_text += f"    df = pd.read_json(f_reader, orient='{orient}',\n"
    func_text += f"       convert_dates = {convert_dates}, \n"
    func_text += f"       precise_float={precise_float}, \n"
    func_text += f"       lines={lines}, \n"
    func_text += "       dtype={{{}}},\n".format(pd_dtype_strs)
    func_text += "       )\n"
    func_text += "    bodo.ir.connector.cast_float_to_nullable(df, df_typeref_2)\n"
    for s_cname, cname in zip(sanitized_cnames, col_names):
        func_text += "    if len(df) > 0:\n"
        func_text += "        {} = df['{}'].values\n".format(s_cname, cname)
        func_text += "    else:\n"
        func_text += "        {} = np.array([])\n".format(s_cname)
    func_text += "  return ({},)\n".format(", ".join(sc for sc in sanitized_cnames))
    glbls = globals()  # TODO: fix globals after Numba's #3355 is resolved
    glbls.update(
        {
            "bodo": bodo,
            "pd": pd,
            "np": np,
            "check_java_installation": check_java_installation,
            "df_typeref": bodo.DataFrameType(
                tuple(col_typs), bodo.RangeIndexType(None), tuple(col_names)
            ),
            "get_storage_options_pyobject": get_storage_options_pyobject,
        }
    )
    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    json_reader_py = loc_vars["json_reader_py"]

    # TODO: no_cpython_wrapper=True crashes for some reason
    jit_func = numba.njit(json_reader_py)
    compiled_funcs.append(jit_func)
    return jit_func
