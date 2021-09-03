# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Implementation of pd.read_sql in BODO.
We piggyback on the pandas implementation. Future plan is to have a faster
version for this task.
"""

import numba
import pandas as pd  # noqa
from numba.core import ir, ir_utils, typeinfer
from numba.core.ir_utils import compile_to_numba_ir, replace_arg_nodes

import bodo
import bodo.ir.connector
from bodo import objmode  # noqa
from bodo.ir.csv_ext import _get_dtype_str
from bodo.libs.distributed_api import bcast, bcast_scalar
from bodo.libs.str_ext import string_type
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.utils.typing import BodoError
from bodo.utils.utils import sanitize_varname

MPI_ROOT = 0


class SqlReader(ir.Stmt):
    def __init__(
        self,
        sql_request,
        connection,
        df_out,
        df_colnames,
        out_vars,
        out_types,
        loc,
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
    sql_node, array_dists, typemap, calltypes, typingctx, targetctx
):
    parallel = False
    if array_dists is not None:
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
        sql_node.df_colnames,
        sql_node.out_types,
        typingctx,
        targetctx,
        sql_node.connection.split(":")[0],  # capture db type from connection string
        parallel,
    )

    f_block = compile_to_numba_ir(
        sql_impl,
        {"_sql_reader_py": sql_reader_py, "bcast_scalar": bcast_scalar, "bcast": bcast},
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=(string_type, string_type),
        typemap=typemap,
        calltypes=calltypes,
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


numba.parfors.array_analysis.array_analysis_extensions[
    SqlReader
] = bodo.ir.connector.connector_array_analysis
distributed_analysis.distributed_analysis_extensions[
    SqlReader
] = bodo.ir.connector.connector_distributed_analysis
typeinfer.typeinfer_extensions[SqlReader] = bodo.ir.connector.connector_typeinfer
ir_utils.visit_vars_extensions[SqlReader] = bodo.ir.connector.visit_vars_connector
ir_utils.remove_dead_extensions[SqlReader] = remove_dead_sql
numba.core.analysis.ir_extension_usedefs[
    SqlReader
] = bodo.ir.connector.connector_usedefs
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


@numba.njit
def sqlalchemy_check():
    with numba.objmode():
        sqlalchemy_check_()


def sqlalchemy_check_():  # pragma: no cover
    try:
        import sqlalchemy  # noqa
    except ImportError:
        message = (
            "Using URI string without sqlalchemy installed."
            " sqlalchemy can be installed by calling"
            " 'conda install -c conda-forge sqlalchemy'."
        )
        raise BodoError(message)


def _gen_sql_reader_py(col_names, col_typs, typingctx, targetctx, db_type, parallel):
    sanitized_cnames = [sanitize_varname(c) for c in col_names]
    typ_strs = [
        "{}='{}'".format(s_cname, _get_dtype_str(t))
        for s_cname, t in zip(sanitized_cnames, col_typs)
    ]
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
        func_text += "  sqlalchemy_check()\n"
        func_text += "  rank = bodo.libs.distributed_api.get_rank()\n"
        func_text += "  n_pes = bodo.libs.distributed_api.get_size()\n"
        func_text += "  with objmode({}):\n".format(", ".join(typ_strs))
        func_text += "    list_df_block = []\n"
        func_text += "    block_size = 50000\n"
        func_text += "    iter = 0\n"
        func_text += "    while(True):\n"
        func_text += "      offset = (iter * n_pes + rank) * block_size\n"
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
        func_text += "  sqlalchemy_check()\n"
        if parallel:
            func_text += "  rank = bodo.libs.distributed_api.get_rank()\n"
            func_text += '  with objmode(nb_row="int64"):\n'
            func_text += f"    if rank == {MPI_ROOT}:\n"
            func_text += (
                "      sql_cons = 'select count(*) from (' + sql_request + ') x'\n"
            )
            func_text += "      frame = pd.read_sql(sql_cons, conn)\n"
            func_text += "      nb_row = frame.iat[0,0]\n"
            func_text += "    else:\n"
            func_text += "      nb_row = 0\n"
            func_text += "  nb_row = bcast_scalar(nb_row)\n"
            func_text += "  with objmode({}):\n".format(", ".join(typ_strs))
            func_text += "    offset, limit = bodo.libs.distributed_api.get_start_count(nb_row)\n"
            # Snowflake doesn't provide consistent output in different processes with
            # LIMIT/OFFSET case and row_number() is recommended instead
            if db_type == "snowflake":
                func_text += "    start_num = offset + 1\n"
                func_text += "    end_num = offset + limit + 1\n"
                func_text += f"    sql_cons = 'select {', '.join(col_names)} from (select row_number() over (order by 1) as row_num, * from (' + sql_request + ')) where row_num >=' + str(start_num) + ' and row_num <' + str(end_num)\n"
                str_cols = [f"'{c}'" for c in col_names]
                func_text += f"    col_names = ({','.join(str_cols)}{',' if len(str_cols) > 0 else ''})\n"
                func_text += "    df_ret = read_snowflake(sql_cons, conn, col_names)\n"
            else:
                func_text += f"    sql_cons = 'select {', '.join(col_names)} from (' + sql_request + ') x LIMIT ' + str(limit) + ' OFFSET ' + str(offset)\n"
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


def read_snowflake(sql_cons, conn, expected_colnames):
    """read query from Snowflake using its connector"""
    try:
        import snowflake.connector
        import sqlalchemy
    except ImportError:
        raise BodoError(
            "pd.read_sql(): reading from Snowflake requires snowflake connector. Install using 'conda install snowflake-sqlalchemy snowflake-connector-python -c bodo.ai -c conda-forge'"
        )

    # get connection parameters from connection string
    # some paramters could be part of address or extra paramters
    # "snowflake://user:pw@account/db/schema?warehouse=XL_WH"
    # "snowflake://account/?user=user&password=pw&database=db&schema=schema&warehouse=XL_WH"
    conn_url = sqlalchemy.engine.url._parse_rfc1738_args(conn)
    params = {}
    if conn_url.username:
        params["user"] = conn_url.username
    if conn_url.password:
        params["password"] = conn_url.password
    if conn_url.host:
        params["account"] = conn_url.host
    if conn_url.database:
        # sqlalchemy reads "db/schema" as just database name so need to split schema
        db = conn_url.database.split("/")[0]
        params["database"] = db
        if "/" in conn_url.database:
            params["schema"] = conn_url.database.split("/")[1]
    if conn_url.port:
        params["port"]: conn_url.port
    params.update(conn_url.query)

    ctx = snowflake.connector.connect(**params)
    cur = ctx.cursor()
    cur.execute(sql_cons)
    df = cur.fetch_pandas_all()
    # sqlalchemy uses lowercase for case insensitive cases but Snowflake uses upper case
    # case sensitive cases use quotes
    # see https://github.com/snowflakedb/snowflake-sqlalchemy
    #
    # To handle this, we check if the column names are uppercase and the expected datatype
    # is lowercase. If so we convert the columns. TODO: Make sure this is fully robust/we
    # don't handle case sensitivity in the Snowflake connector properly.
    df.columns = [
        expected_colnames[i] if s.isupper() else s for i, s in enumerate(df.columns)
    ]
    return df
