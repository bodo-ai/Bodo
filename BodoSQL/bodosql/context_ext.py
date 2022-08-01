"""Bodo extensions to support BodoSQLContext inside JIT functions.
Assumes an immutable context where table names and dataframes are not modified inplace,
which allows typing and optimization.
"""
import re

import bodo
import numba
import numpy as np
import pandas as pd
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.libs.distributed_api import bcast_scalar
from bodo.utils.typing import BodoError, get_overload_const
from numba.core import cgutils, types
from numba.extending import (
    intrinsic,
    make_attribute_wrapper,
    models,
    overload,
    overload_method,
    register_model,
    typeof_impl,
)

from bodosql.bodosql_types.table_path import TablePathType
from bodosql.context import (
    NAMED_PARAM_TABLE_NAME,
    BodoSQLContext,
    BodoSqlSchemaClass,
    RelationalAlgebraGeneratorClass,
    compute_df_types,
    generate_used_table_func_text,
    get_used_tables_from_generator,
    intialize_database,
)
from bodosql.utils import java_error_to_msg


class BodoSQLContextType(types.Type):
    """Data type for compiling BodoSQLContext.
    Requires table names and dataframe types.
    """

    def __init__(self, names, dataframes):
        if not (isinstance(names, tuple) and all(isinstance(v, str) for v in names)):
            raise BodoError("BodoSQLContext(): 'table' keys must be constant strings")
        if not (
            isinstance(dataframes, tuple)
            and all(isinstance(v, (DataFrameType, TablePathType)) for v in dataframes)
        ):
            raise BodoError(
                "BodoSQLContext(): 'table' values must be DataFrames or TablePaths"
            )
        self.names = names
        self.dataframes = dataframes
        super(BodoSQLContextType, self).__init__(
            name=f"BodoSQLContextType({names}, {dataframes})"
        )


@typeof_impl.register(BodoSQLContext)
def typeof_bodo_sql(val, c):
    return BodoSQLContextType(
        tuple(val.tables.keys()), tuple(numba.typeof(v) for v in val.tables.values())
    )


@register_model(BodoSQLContextType)
class BodoSQLContextModel(models.StructModel):
    """store BodoSQLContext's tables as a tuple of dataframes"""

    def __init__(self, dmm, fe_type):
        members = [
            ("dataframes", types.BaseTuple.from_types(fe_type.dataframes)),
        ]
        super(BodoSQLContextModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(BodoSQLContextType, "dataframes", "dataframes")


def lower_init_sql_context(context, builder, signature, args):
    """lowering code to initialize a BodoSQLContextType"""
    sql_context_type = signature.return_type
    sql_ctx_struct = cgutils.create_struct_proxy(sql_context_type)(context, builder)
    context.nrt.incref(builder, signature.args[1], args[1])
    sql_ctx_struct.dataframes = args[1]
    return sql_ctx_struct._getvalue()


@intrinsic
def init_sql_context(typingctx, names_type, dataframes_type=None):
    """Create a BodoSQLContext given table names and dataframes."""
    table_names = tuple(get_overload_const(names_type))
    n_tables = len(names_type.types)
    assert len(dataframes_type.types) == n_tables
    sql_ctx_type = BodoSQLContextType(table_names, tuple(dataframes_type.types))
    return sql_ctx_type(names_type, dataframes_type), lower_init_sql_context


# enable dead call elimination for init_sql_context()
bodo.utils.transform.no_side_effect_call_tuples.add((init_sql_context,))


@overload(BodoSQLContext, inline="always", no_unliteral=True)
def bodo_sql_context_overload(tables):
    """constructor for creating BodoSQLContext"""
    # bodo untyped pass transforms const dict to tuple with sentinel in first element
    assert isinstance(tables, types.BaseTuple) and tables.types[
        0
    ] == types.StringLiteral("__bodo_tup"), "BodoSQLContext(): invalid tables input"
    assert len(tables.types) % 2 == 1, "invalid const dict tuple structure"
    n_dfs = (len(tables.types) - 1) // 2
    names = [t.literal_value for t in tables.types[1 : n_dfs + 1]]
    df_args = ", ".join("tables[{}]".format(i) for i in range(n_dfs + 1, 2 * n_dfs + 1))

    df_args = "({}{})".format(df_args, "," if len(names) == 1 else "")
    name_args = ", ".join(f"'{c}'" for c in names)
    names_tup = "({}{})".format(name_args, "," if len(names) == 1 else "")

    func_text = (
        f"def impl(tables):\n  return init_sql_context({names_tup}, {df_args})\n"
    )
    loc_vars = {}
    _global = {"init_sql_context": init_sql_context}
    exec(func_text, _global, loc_vars)
    impl = loc_vars["impl"]
    return impl


def _gen_pd_func_text_and_lowered_globals(
    bodo_sql_context_type, sql_str, param_keys, param_values, is_optimized=True
):
    """
    Helper function called by _gen_pd_func_for_query and _gen_pd_func_str_for_query.

    Generates the func_text by calling our calcite application on rank 0. Throws a BodoError if
    it encounters an error.

    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    if sql_str.strip() == "":
        raise BodoError("BodoSQLContext passed empty query string")

    # Since we're only creating the func text on rank 0, we need to broadcast to
    # the other ranks if we encounter an error. In the case that we encounter an error,
    # func_text will be replaced with the error message.
    failed = False
    func_text_or_error_msg = ""
    globalsToLower = ()
    try:
        orig_bodo_types, df_types = compute_df_types(
            bodo_sql_context_type.dataframes, True
        )
    except Exception as e:
        raise BodoError(
            f"Unable to determine one or more DataFrames in BodoSQL query: {e}"
        )
    if bodo.get_rank() == 0:
        # This outermost try except should normally never be invoked, but it's here for safety
        # So the other ranks don't hang forever if we encounter an unexpected runtime error
        try:
            table_names = bodo_sql_context_type.names

            name_to_df_type_dict = dict(zip(table_names, df_types))
            name_to_bodo_type_dict = dict(zip(table_names, orig_bodo_types))
            db, table_conversions = intialize_database(
                name_to_df_type_dict, name_to_bodo_type_dict, (param_keys, param_values)
            )
            schema = BodoSqlSchemaClass(db)
            generator = RelationalAlgebraGeneratorClass(schema, NAMED_PARAM_TABLE_NAME)
            try:
                if is_optimized:
                    pd_code = str(generator.getPandasString(sql_str))
                else:
                    pd_code = str(generator.getPandasStringUnoptimized(sql_str))
                used_tables = get_used_tables_from_generator(generator)
                # Convert to tuple of string tuples, to allow bcast to work
                globalsToLower = tuple(
                    [
                        (str(k), str(v))
                        for k, v in generator.getLoweredGlobalVariables().items()
                    ]
                )
            except Exception as e:
                # Raise BodoError outside except to avoid stack trace
                func_text_or_error_msg = (
                    f"Unable to parse SQL Query. Error message: {java_error_to_msg(e)}"
                )
                failed = True
            if not failed:
                args = ",".join(("bodo_sql_context",) + param_keys)
                func_text_or_error_msg = f"def impl({args}):\n"
                pd_func_text = f"{pd_code}\n"
                func_text_or_error_msg += generate_used_table_func_text(
                    table_names,
                    used_tables,
                    orig_bodo_types,
                    table_conversions,
                    pd_func_text,
                    True,
                )
        except Exception as e:
            func_text_or_error_msg = (
                f"Unable to parse SQL Query due to unexpected runtime error: \n{str(e)}"
            )
            failed = True
    failed = bcast_scalar(failed)
    func_text_or_error_msg = bcast_scalar(func_text_or_error_msg)
    if failed:
        raise bodo.utils.typing.BodoError(func_text_or_error_msg)
    globalsToLower = comm.bcast(globalsToLower)

    # Convert the globalsToLower from a list of tuples of strings to a dict of string varname -> value
    outGlobalsDict = {}
    # convert the global map list of tuples of string varname and string value, to a map of string varname -> python value.
    for varname, str_value in globalsToLower:
        locs = {}
        exec(
            f"value = {str_value}",
            {
                "ColNamesMetaType": bodo.utils.typing.ColNamesMetaType,
                "MetaType": bodo.utils.typing.MetaType,
                "numba": numba,
                "bodo": bodo,
            },
            locs,
        )
        outGlobalsDict[varname] = locs["value"]
    return func_text_or_error_msg, outGlobalsDict


def _gen_pd_func_and_glbls_for_query(
    bodo_sql_context_type, sql_str, param_keys, param_values
):
    """Generate a Pandas function for query given the data type of SQL context.
    Used in Bodo typing pass to handle BodoSQLContext.sql() calls
    """
    import bodosql

    func_text, glblsToLower = _gen_pd_func_text_and_lowered_globals(
        bodo_sql_context_type, sql_str, param_keys, param_values
    )

    loc_vars = {}
    exec(
        func_text,
        {"pd": pd, "np": np, "bodo": bodo, "re": re, "bodosql": bodosql},
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl, glblsToLower


@overload_method(BodoSQLContextType, "sql", inline="always", no_unliteral=True)
def overload_sql(bodo_sql_context, sql_str, param_dict=None):
    """BodoSQLContextType.sql() should be handled in bodo typing pass since the
    generated code cannot be handled in regular overloads
    (requires Bodo's untyped pass and typing pass)
    """
    bodo.utils.typing.raise_bodo_error("Invalid BodoSQLContext.sql() call")


def _gen_pd_func_str_for_query(
    bodo_sql_context_type, sql_str, param_keys, param_values
):
    """Generate a function that returns the string of code that would be generated
    for the query given the data type of SQL context.
    Used in Bodo's typing pass to handle BodoSQLContext.convert_to_pandas() calls
    """

    # Don't need globals here, just need the func_text
    returned_func_text, globalsToLower = _gen_pd_func_text_and_lowered_globals(
        bodo_sql_context_type, sql_str, param_keys, param_values
    )

    # In this case, since the func_text is not going to be executed, we just
    # replace the lowered globals with the original values
    for k, v in globalsToLower.items():
        returned_func_text = returned_func_text.replace(str(k), str(v))

    executed_func_text = f"def impl(bodo_sql_context):\n"
    # This doesn't work if we have triple quotes within the generated code
    # This currently isn't an issue, but I'm making note of it just in case
    executed_func_text += f'  return """{returned_func_text}"""'
    loc_vars = dict()
    imports = dict()
    exec(
        executed_func_text,
        imports,
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl, {}


@overload_method(
    BodoSQLContextType, "convert_to_pandas", inline="always", no_unliteral=True
)
def overload_convert_to_pandas(bodo_sql_context, sql_str, param_dict=None):
    """BodoSQLContextType.convert_to_pandas() should be handled in bodo typing pass since the
    generated code cannot be handled in regular overloads
    (requires Bodo's untyped pass and typing pass)
    """
    bodo.utils.typing.raise_bodo_error(
        "Invalid BodoSQLContext.convert_to_pandas() call"
    )


def _gen_pd_func_and_globals_for_unoptimized_query(
    bodo_sql_context_type, sql_str, param_keys, param_values
):
    """Generate a Pandas function for query given the data type of SQL context.
    Used in Bodo typing pass to handle BodoSQLContext._test_sql_unoptimized() calls.
    This function generates code without performing optimizations for testing
    coverage of operations that tend to have simple cases optimized out.
    """
    import bodosql

    func_text, globalsToLower = _gen_pd_func_text_and_lowered_globals(
        bodo_sql_context_type, sql_str, param_keys, param_values, is_optimized=False
    )

    loc_vars = {}
    exec(
        func_text,
        {"pd": pd, "np": np, "bodo": bodo, "re": re, "bodosql": bodosql},
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl, globalsToLower


@overload_method(
    BodoSQLContextType, "_test_sql_unoptimized", inline="always", no_unliteral=True
)
def overload_test_sql_unoptimized(bodo_sql_context, sql_str):
    """BodoSQLContextType._test_sql_unoptimized() should be handled in bodo typing pass since the
    generated code cannot be handled in regular overloads
    (requires Bodo's untyped pass and typing pass)
    """
    bodo.utils.typing.raise_bodo_error("Invalid BodoSQLContext.sql() call")
