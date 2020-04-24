# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Implementation of pd.read_sql in BODO.
We piggyback on the pandas implementation. Future plan is to have a faster
version for this task.
"""
from collections import defaultdict
import numba
from numba import typeinfer, ir, ir_utils, config, types, cgutils
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from numba.typing.templates import signature
from numba.extending import overload, intrinsic, register_model, models, box
from numba.ir_utils import (
    visit_vars_inner,
    replace_vars_inner,
    compile_to_numba_ir,
    replace_arg_nodes,
)
import bodo
from bodo.transforms import distributed_pass, distributed_analysis
from bodo.hiframes.datetime_date_ext import DatetimeDateType
from bodo.utils.utils import debug_prints
from bodo.transforms.distributed_analysis import Distribution
from bodo.libs.str_ext import string_type
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.timsort import copyElement_tup, getitem_arr_tup
from bodo.utils.utils import sanitize_varname
from bodo.ir.csv_ext import _get_dtype_str
from bodo.libs.distributed_api import bcast_scalar, bcast
from bodo import objmode
import pandas as pd
import numpy as np

from bodo.hiframes.pd_categorical_ext import PDCategoricalDtype, CategoricalArray

MPI_ROOT = 0


class SqlReader(ir.Stmt):
    def __init__(
        self, sql_request, connection, df_out, df_colnames, out_vars, out_types, loc,
    ):
        self.connector_typ = "sql"
        self.sql_request = sql_request
        self.connection = connection
        self.df_out = df_out  # used only for printing
        self.df_colnames = df_colnames
        self.out_vars = out_vars
        self.out_types = out_types
        self.loc = loc

    def __repr__(self):  # pragma: no cover
        return "{} = ReadSql(sql_request={}, connection={}, col_names={}, types={}, vars={})".format(
            self.df_out,
            self.sql_request,
            self.connection,
            self.df_colnames,
            self.out_types,
            self.out_vars,
        )


def remove_dead_sql(
    sql_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):
    # TODO
    new_df_colnames = []
    new_out_vars = []
    new_out_types = []

    for i, col_var in enumerate(sql_node.out_vars):
        if col_var.name in lives:
            new_df_colnames.append(sql_node.df_colnames[i])
            new_out_vars.append(sql_node.out_vars[i])
            new_out_types.append(sql_node.out_types[i])

    sql_node.df_colnames = new_df_colnames
    sql_node.out_vars = new_out_vars
    sql_node.out_types = new_out_types

    if len(sql_node.out_vars) == 0:
        return None

    return sql_node


def sql_distributed_run(
    sql_node, array_dists, typemap, calltypes, typingctx, targetctx, dist_pass
):
    parallel = True
    for v in sql_node.out_vars:
        if (
            array_dists[v.name] != distributed_pass.Distribution.OneD
            and array_dists[v.name] != distributed_pass.Distribution.OneD_Var
        ):
            parallel = False

    n_cols = len(sql_node.out_vars)
    arg_names = ", ".join("arr" + str(i) for i in range(n_cols))
    func_text = "def sql_impl(sql_request, conn):\n"
    func_text += "    ({},) = _sql_reader_py(sql_request, conn)\n".format(arg_names)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    sql_impl = loc_vars["sql_impl"]

    sql_reader_py = _gen_sql_reader_py(
        sql_node.df_colnames, sql_node.out_types, typingctx, targetctx, parallel,
    )

    f_block = compile_to_numba_ir(
        sql_impl,
        {"_sql_reader_py": sql_reader_py, "bcast_scalar": bcast_scalar, "bcast": bcast},
        typingctx,
        (string_type, string_type),
        typemap,
        calltypes,
    ).blocks.popitem()[1]
    replace_arg_nodes(
        f_block,
        [
            ir.Const(sql_node.sql_request, sql_node.loc),
            ir.Const(sql_node.connection, sql_node.loc),
        ],
    )
    nodes = f_block.body[:-3]
    for i in range(len(sql_node.out_vars)):
        nodes[-len(sql_node.out_vars) + i].target = sql_node.out_vars[i]

    return nodes


numba.array_analysis.array_analysis_extensions[
    SqlReader
] = bodo.ir.connector.connector_array_analysis
distributed_analysis.distributed_analysis_extensions[
    SqlReader
] = bodo.ir.connector.connector_distributed_analysis
typeinfer.typeinfer_extensions[SqlReader] = bodo.ir.connector.connector_typeinfer
ir_utils.visit_vars_extensions[SqlReader] = bodo.ir.connector.visit_vars_connector
ir_utils.remove_dead_extensions[SqlReader] = remove_dead_sql
numba.analysis.ir_extension_usedefs[SqlReader] = bodo.ir.connector.connector_usedefs
ir_utils.copy_propagate_extensions[SqlReader] = bodo.ir.connector.get_copies_connector
ir_utils.apply_copy_propagate_extensions[
    SqlReader
] = bodo.ir.connector.apply_copies_connector
ir_utils.build_defs_extensions[
    SqlReader
] = bodo.ir.connector.build_connector_definitions
distributed_pass.distributed_run_extensions[SqlReader] = sql_distributed_run


# XXX: temporary fix pending Numba's #3378
# keep the compiled functions around to make sure GC doesn't delete them and
# the reference to the dynamic function inside them
# (numba/lowering.py:self.context.add_dynamic_addr ...)
compiled_funcs = []


def _gen_sql_reader_py(col_names, col_typs, typingctx, targetctx, parallel):
    sanitized_cnames = [sanitize_varname(c) for c in col_names]
    date_inds = ", ".join(
        str(i) for i, t in enumerate(col_typs) if t.dtype == types.NPDatetime("ns")
    )
    typ_strs = [
        "{}='{}'".format(s_cname, _get_dtype_str(t))
        for s_cname, t in zip(sanitized_cnames, col_typs)
    ]
    rank = bodo.libs.distributed_api.get_rank()
    n_pes = bodo.libs.distributed_api.get_size()
    # Method below is incorrect if the data is required to be ordered.
    # It is left here as historical record.
    #
    # Algorithm is following:
    # ---We download blocks one by one.
    # ---Each MPI computational process downloads a block according to its
    #    range.
    # ---Termination happens when the size of blocks is 0.
    # ---Different MPI process may have different number of blocks.
    #    (but the number will differ by at most 1 between MPI nodes)
    if bodo.sql_access_method == "multiple_access_by_block":
        func_text = "def sql_reader_py(sql_request,conn):\n"
        func_text += "  with objmode({}):\n".format(", ".join(typ_strs))
        func_text += "    list_df_block = []\n"
        func_text += "    block_size = 50000\n"
        func_text += "    iter = 0\n"
        func_text += "    while(True):\n"
        func_text += "      offset = (iter * {} + {}) * block_size\n".format(
            n_pes, rank
        )
        func_text += "      sql_cons = 'select * from (' + sql_request + ') x LIMIT ' + str(block_size) + ' OFFSET ' + str(offset)\n"
        func_text += "      df_block = pd.read_sql(sql_cons, conn)\n"
        func_text += "      if df_block.size == 0:\n"
        func_text += "        break\n"
        func_text += "      list_df_block.append(df_block)\n"
        func_text += "      iter += 1\n"
        func_text += "    df_ret = pd.concat(list_df_block)\n"
        # We assumed here that sanitized_cnames and col_names are list of strings.
        for s_cname, cname in zip(sanitized_cnames, col_names):
            func_text += "    {} = df_ret['{}'].values\n".format(s_cname, cname)
        func_text += "  return ({},)\n".format(", ".join(sc for sc in sanitized_cnames))
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
    if bodo.sql_access_method == "multiple_access_nb_row_first":
        func_text = "def sql_reader_py(sql_request, conn):\n"
        if parallel:
            func_text += '  with objmode(nb_row="int64"):\n'
            if rank == MPI_ROOT:
                func_text += (
                    "    sql_cons = 'select count(*) from (' + sql_request + ') x'\n"
                )
                func_text += "    frame = pd.read_sql(sql_cons, conn)\n"
                func_text += "    nb_row = frame.iat[0,0]\n"
            else:
                func_text += "    nb_row = 0\n"
            func_text += "  nb_row = bcast_scalar(nb_row)\n"
            func_text += "  with objmode({}):\n".format(", ".join(typ_strs))
            func_text += "    rank = {}\n".format(rank)
            func_text += "    n_pes = {}\n".format(n_pes)
            func_text += "    offset, limit = bodo.libs.distributed_api.get_start_count(nb_row)\n"
            func_text += "    sql_cons = 'select * from (' + sql_request + ') x LIMIT ' + str(limit) + ' OFFSET ' + str(offset)\n"
            func_text += "    df_ret = pd.read_sql(sql_cons, conn)\n"
        else:
            func_text += "  with objmode({}):\n".format(", ".join(typ_strs))
            func_text += "    df_ret = pd.read_sql(sql_request, conn)\n"
        # We assumed that sanitized_cnames and col_names are list of strings
        for s_cname, cname in zip(sanitized_cnames, col_names):
            func_text += "    {} = df_ret['{}'].values\n".format(s_cname, cname)
        func_text += "  return ({},)\n".format(", ".join(sc for sc in sanitized_cnames))

    glbls = globals()  # TODO: fix globals after Numba's #3355 is resolved
    # {'objmode': objmode,
    # 'pd': pd, 'np': np}
    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    sql_reader_py = loc_vars["sql_reader_py"]

    # TODO: no_cpython_wrapper=True crashes for some reason
    jit_func = numba.njit(sql_reader_py)
    compiled_funcs.append(jit_func)
    return jit_func
