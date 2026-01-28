"""Bodo extensions to support BodoSQLContext inside JIT functions.
Assumes an immutable context where table names and DataFrames are not modified inplace,
which allows typing and optimization.
"""

import datetime
import re
import time
import warnings
from typing import Any

import numba
import numpy as np
import pandas as pd
from numba.core import cgutils, ir, types
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)

import bodo
import bodo.hiframes
import bodo.hiframes.pd_multi_index_ext
import bodo.io.iceberg.merge_into  # noqa
import bodo.io.iceberg.read_compilation
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.io.utils import parse_dbtype
from bodo.libs.distributed_api import bcast_scalar
from bodo.utils.typing import (
    BodoError,
    NotConstant,
    assert_bodo_error,
    dtype_to_array_type,
    get_overload_const,
    get_overload_const_str,
    is_overload_constant_str,
    is_overload_none,
    raise_bodo_error,
)
from bodosql.bodosql_types.database_catalog_ext import DatabaseCatalogType
from bodosql.bodosql_types.table_path_ext import TablePathType
from bodosql.context import (
    DYNAMIC_PARAM_ARG_PREFIX,
    NAMED_PARAM_ARG_PREFIX,
    BodoSQLContext,
    SqlTypeEnum,
    create_java_dynamic_parameter_type_list,
    create_java_named_parameter_type_map,
    initialize_schema,
    update_schema,
)
from bodosql.imported_java_classes import (
    JavaEntryPoint,
)
from bodosql.utils import BodoSQLWarning, error_to_string


class BodoSQLContextType(types.Type):
    """Data type for compiling BodoSQLContext.
    Requires table names and DataFrame types.
    """

    def __init__(self, names, dataframes, estimated_row_counts, catalog, default_tz):
        if not (isinstance(names, tuple) and all(isinstance(v, str) for v in names)):
            raise BodoError("BodoSQLContext(): 'table' keys must be constant strings")
        if not (
            isinstance(dataframes, tuple)
            and all(isinstance(v, (DataFrameType, TablePathType)) for v in dataframes)
        ):
            raise BodoError(
                "BodoSQLContext(): 'table' values must be DataFrames or TablePaths"
            )
        if not (isinstance(catalog, DatabaseCatalogType) or is_overload_none(catalog)):
            raise BodoError(
                "BodoSQLContext(): 'catalog' must be a bodosql.DatabaseCatalog if provided"
            )
        if not (
            isinstance(estimated_row_counts, tuple)
            and all(
                isinstance(row_count, int) or row_count is None
                for row_count in estimated_row_counts
            )
        ):
            raise BodoError(
                "BodoSQLContext(): 'estimated_row_counts' must be a tuple of int or None for each table."
            )
        self.names = names
        self.dataframes = dataframes
        self.estimated_row_counts = estimated_row_counts
        # Map None to types.none to use the type in the data model.
        self.default_tz = default_tz
        self.catalog_type = types.none if is_overload_none(catalog) else catalog
        super().__init__(
            name=f"BodoSQLContextType({names}, {dataframes}, {estimated_row_counts}, {catalog}, {default_tz})"
        )


@typeof_impl.register(BodoSQLContext)
def typeof_bodo_sql(val, c):
    dataframes = val.tables.values()
    return BodoSQLContextType(
        tuple(val.tables.keys()),
        tuple(numba.typeof(df) for df in dataframes),
        # TODO(njriasan): Handle distributed code.
        tuple([len(df) if isinstance(df, pd.DataFrame) else None for df in dataframes]),
        numba.typeof(val.catalog),
        val.default_tz,
    )


@register_model(BodoSQLContextType)
class BodoSQLContextModel(models.StructModel):
    """store BodoSQLContext's tables as a tuple of dataframes"""

    def __init__(self, dmm, fe_type):
        members = [
            ("dataframes", types.BaseTuple.from_types(fe_type.dataframes)),
            ("catalog", fe_type.catalog_type),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(BodoSQLContextType, "dataframes", "dataframes")
make_attribute_wrapper(BodoSQLContextType, "catalog", "catalog")


@overload_attribute(BodoSQLContextType, "default_tz", inline="always")
def overload_default_tz(bc):
    tz = bc.default_tz

    def impl(bc):  # pragma: no cover
        return tz

    return impl


def lower_init_sql_context(context, builder, signature, args):
    """lowering code to initialize a BodoSQLContextType"""
    sql_context_type = signature.return_type
    sql_ctx_struct = cgutils.create_struct_proxy(sql_context_type)(context, builder)
    context.nrt.incref(builder, signature.args[1], args[1])
    sql_ctx_struct.dataframes = args[1]
    sql_context_type.catalog = args[2]
    return sql_ctx_struct._getvalue()


@box(BodoSQLContextType)
def box_bodosql_context(typ, val, c):
    """
    Boxes a BodoSQLContext into a Python value.
    """
    # Create a dictionary for python
    py_dict_obj = c.pyapi.dict_new(len(typ.names))
    bodosql_context_struct = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    dataframes = bodosql_context_struct.dataframes
    for i, name in enumerate(typ.names):
        df = c.builder.extract_value(dataframes, i)
        c.context.nrt.incref(c.builder, typ.dataframes[i], df)
        df_obj = c.pyapi.from_native_value(typ.dataframes[i], df, c.env_manager)
        c.pyapi.dict_setitem_string(py_dict_obj, name, df_obj)
        c.pyapi.decref(df_obj)

    # Box the catalog if it exists
    if is_overload_none(typ.catalog_type):
        catalog_obj = c.pyapi.make_none()
    else:
        c.context.nrt.incref(
            c.builder, typ.catalog_type, bodosql_context_struct.catalog
        )
        catalog_obj = c.pyapi.from_native_value(
            typ.catalog_type, bodosql_context_struct.catalog, c.env_manager
        )

    mod_name = c.context.insert_const_string(c.builder.module, "bodosql")
    bodosql_class_obj = c.pyapi.import_module(mod_name)
    res = c.pyapi.call_method(
        bodosql_class_obj, "BodoSQLContext", (py_dict_obj, catalog_obj)
    )
    c.pyapi.decref(bodosql_class_obj)
    c.pyapi.decref(py_dict_obj)
    c.pyapi.decref(catalog_obj)
    c.context.nrt.decref(c.builder, typ, val)
    return res


@unbox(BodoSQLContextType)
def unbox_bodosql_context(typ, val, c):
    """
    Unboxes a BodoSQLContext into a native value.
    """
    # Unbox the tables
    py_dfs_obj = c.pyapi.object_getattr_string(val, "tables")
    native_dfs = []
    for i, name in enumerate(typ.names):
        df_obj = c.pyapi.dict_getitem_string(py_dfs_obj, name)
        df_struct = c.pyapi.to_native_value(typ.dataframes[i], df_obj)
        # Set the parent value
        c.pyapi.incref(df_obj)
        df_struct.parent = df_obj
        native_dfs.append(df_struct.value)
        c.pyapi.decref(df_obj)
    c.pyapi.decref(py_dfs_obj)
    df_tuple = c.context.make_tuple(c.builder, types.Tuple(typ.dataframes), native_dfs)
    # Unbox the catalog
    catalog_obj = c.pyapi.object_getattr_string(val, "catalog")
    catalog_value = c.pyapi.to_native_value(typ.catalog_type, catalog_obj).value
    c.pyapi.decref(catalog_obj)
    # Populate the struct
    bodosql_context_struct = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    bodosql_context_struct.dataframes = df_tuple
    bodosql_context_struct.catalog = catalog_value
    return NativeValue(bodosql_context_struct._getvalue())


@intrinsic(prefer_literal=True)
def init_sql_context(typingctx, names_type, dataframes_type, catalog, default_tz):
    """Create a BodoSQLContext given table names and dataframes."""
    table_names = get_overload_const(names_type)
    assert_bodo_error(not isinstance(table_names, NotConstant))
    table_names = tuple(table_names)
    n_tables = len(names_type.types)
    assert len(dataframes_type.types) == n_tables
    # Cannot estimate row counts in compiled code at this time.
    estimated_row_counts = tuple([None] * len(dataframes_type.types))
    sql_ctx_type = BodoSQLContextType(
        table_names,
        tuple(dataframes_type.types),
        estimated_row_counts,
        catalog,
        default_tz,
    )
    return (
        sql_ctx_type(names_type, dataframes_type, catalog, default_tz),
        lower_init_sql_context,
    )


# enable dead call elimination for init_sql_context()
bodo.utils.transform.no_side_effect_call_tuples.add((init_sql_context,))


@overload(BodoSQLContext, inline="always", no_unliteral=True)
def bodo_sql_context_overload(tables, catalog=None, default_tz=None):
    """constructor for creating BodoSQLContext"""
    # bodo untyped pass transforms const dict to tuple with sentinel in first element
    assert isinstance(tables, types.BaseTuple) and tables.types[
        0
    ] == types.StringLiteral("__bodo_tup"), "BodoSQLContext(): invalid tables input"
    assert len(tables.types) % 2 == 1, "invalid const dict tuple structure"
    n_dfs = (len(tables.types) - 1) // 2
    names = [t.literal_value for t in tables.types[1 : n_dfs + 1]]
    df_args = ", ".join(f"tables[{i}]" for i in range(n_dfs + 1, 2 * n_dfs + 1))

    df_args = "({}{})".format(df_args, "," if len(names) == 1 else "")
    name_args = ", ".join(f"'{c}'" for c in names)
    names_tup = "({}{})".format(name_args, "," if len(names) == 1 else "")
    func_text = f"def impl(tables, catalog=None, default_tz=None):\n  return init_sql_context({names_tup}, {df_args}, catalog, default_tz)\n"
    loc_vars = {}
    _global = {"init_sql_context": init_sql_context}
    exec(func_text, _global, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_method(BodoSQLContextType, "add_or_replace_view", no_unliteral="True")
def overload_bodosql_context_add_or_replace_view(bc, name, table):
    if not is_overload_constant_str(name):
        raise_bodo_error(
            "BodoSQLContext.add_or_replace_view(): 'name' must be a constant string"
        )
    name = get_overload_const_str(name)
    if not isinstance(table, (bodo.types.DataFrameType, TablePathType)):
        raise BodoError(
            "BodoSQLContext.add_or_replace_view(): 'table' must be a DataFrameType or TablePathType"
        )
    new_names = []
    new_dataframes = []
    for i, old_name in enumerate(bc.names):
        if old_name != name:
            new_names.append(f"'{old_name}'")
            new_dataframes.append(f"bc.dataframes[{i}]")
    new_names.append(f"'{name}'")
    new_dataframes.append("table")
    comma_sep_names = ", ".join(new_names)
    comma_sep_dfs = ", ".join(new_dataframes)
    func_text = "def impl(bc, name, table):\n"
    func_text += f"  return init_sql_context(({comma_sep_names}, ), ({comma_sep_dfs}, ), bc.catalog, bc.default_tz)\n"
    loc_vars = {}
    _global = {"init_sql_context": init_sql_context}
    exec(func_text, _global, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_method(BodoSQLContextType, "remove_view", no_unliteral="True")
def overload_bodosql_context_remove_view(bc, name):
    if not is_overload_constant_str(name):
        raise_bodo_error(
            "BodoSQLContext.remove_view(): 'name' must be a constant string"
        )
    name = get_overload_const_str(name)
    new_names = []
    new_dataframes = []
    found = False
    for i, old_name in enumerate(bc.names):
        if old_name != name:
            new_names.append(f"'{old_name}'")
            new_dataframes.append(f"bc.dataframes[{i}]")
        else:
            found = True
    if not found:
        raise BodoError(
            "BodoSQLContext.remove_view(): 'name' must refer to a registered view"
        )

    comma_sep_names = ", ".join(new_names)
    comma_sep_dfs = ", ".join(new_dataframes)
    func_text = "def impl(bc, name):\n"
    func_text += f"  return init_sql_context(({comma_sep_names}, ), ({comma_sep_dfs}, ), bc.catalog, bc.default_tz)\n"
    loc_vars = {}
    _global = {"init_sql_context": init_sql_context}
    exec(func_text, _global, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_method(BodoSQLContextType, "add_or_replace_catalog")
def overload_add_or_replace_catalog(bc, catalog):
    if not isinstance(catalog, DatabaseCatalogType):
        raise_bodo_error(
            "BodoSQLContext.add_or_replace_catalog(): 'catalog' must be a bodosql.DatabaseCatalog type"
        )
    names = []
    dataframes = []
    for i, name in enumerate(bc.names):
        names.append(f"'{name}'")
        dataframes.append(f"bc.dataframes[{i}]")
    comma_sep_names = ", ".join(names)
    comma_sep_dfs = ", ".join(dataframes)
    func_text = "def impl(bc, catalog):\n"
    func_text += f"  return init_sql_context(({comma_sep_names}, ), ({comma_sep_dfs}, ), catalog, bc.default_tz)\n"
    loc_vars = {}
    _global = {"init_sql_context": init_sql_context}
    exec(func_text, _global, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_method(BodoSQLContextType, "remove_catalog")
def overload_remove_catalog(bc):
    if is_overload_none(bc.catalog_type):
        raise_bodo_error(
            "BodoSQLContext.remove_catalog(): BodoSQLContext must have an existing catalog registered."
        )
    names = []
    dataframes = []
    for i, name in enumerate(bc.names):
        names.append(f"'{name}'")
        dataframes.append(f"bc.dataframes[{i}]")
    comma_sep_names = ", ".join(names)
    comma_sep_dfs = ", ".join(dataframes)
    func_text = "def impl(bc):\n"
    func_text += f"  return init_sql_context(({comma_sep_names}, ), ({comma_sep_dfs}, ), None, bc.default_tz)\n"
    loc_vars = {}
    _global = {"init_sql_context": init_sql_context}
    exec(func_text, _global, loc_vars)
    impl = loc_vars["impl"]
    return impl


def _gen_sql_plan_pd_func_text_and_lowered_globals(
    bodo_sql_context_type: BodoSQLContextType,
    sql_str: str,
    dynamic_param_values: tuple[types.Type],
    named_param_keys: tuple[str],
    named_param_values: tuple[types.Type],
    hide_credentials: bool,
) -> tuple[str, dict[str, Any], str]:
    """
    Helper function called by _gen_pd_func_for_query and _gen_pd_func_str_for_query
    that generates the SQL plan and func_text by calling our calcite application on rank 0.

    Args:
        bodo_sql_context_type (BodoSQLContextType): The BodoSQL context type used to derive
            the necessary configuration for generating the query.
        sql_str (str): The SQL text to parse and execute.
        dynamic_param_values (Tuple[Any]): An N-Tuple of values containing the data for Python variables
            used in SQL passed as bind variables.
        named_param_keys (Tuple[str]): An N-Tuple of keys used to access Python variables in SQL via named parameters.
        named_param_values (Tuple[Any]): An N-Tuple of values containing the data for Python variables
            used in SQL via named parameters.
        hide_credentials (bool): Should credentials be hidden in the generated
            code. This is used when we generate code/plans we want to inspect but
            not run to avoid exposing credentials.
        is_optimized (bool, optional): Should the generated func_text derive
            from an optimized plan. This is set to False for a handful of tests.

    Raises:
        BodoError: If the given SQL cannot be properly processed it raises an error.

    Returns:
        Tuple[str, Dict[str, Any], str]: Returns the generated func_text, a dictionary
            containing the lowered global variables and the SQL plan.
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
    failed = False
    plan = None
    if bodo.get_rank() == 0:
        # This outermost try except should normally never be invoked, but it's here for safety
        # So the other ranks don't hang forever if we encounter an unexpected runtime error
        try:
            table_names = bodo_sql_context_type.names
            schema = initialize_schema()
            verbose_level = bodo.user_logging.get_verbose_level()
            tracing_level = bodo.tracing_level
            if bodo_sql_context_type.catalog_type != types.none:
                catalog_obj = bodo_sql_context_type.catalog_type.get_java_object()
            else:
                catalog_obj = None
            if bodo_sql_context_type.default_tz is None or isinstance(
                bodo_sql_context_type.default_tz, types.NoneType
            ):
                default_tz_str = None
            else:
                default_tz_str = bodo_sql_context_type.default_tz.literal_value
            generator = JavaEntryPoint.buildRelationalAlgebraGenerator(
                catalog_obj,
                schema,
                bodo.bodosql_use_streaming_plan,
                verbose_level,
                tracing_level,
                bodo.bodosql_streaming_batch_size,
                hide_credentials,
                bodo.enable_snowflake_iceberg,
                bodo.enable_timestamp_tz,
                bodo.enable_streaming_sort,
                bodo.enable_streaming_sort_limit_offset,
                bodo.bodo_sql_style,
                bodo.bodosql_full_caching,
                bodo.prefetch_sf_iceberg,
                default_tz_str,
            )
        except Exception as e:
            # Raise BodoError outside except to avoid stack trace
            func_text_or_error_msg = f"Unable to initialize BodoSQL Tables when parsing SQL Query. Error message: {error_to_string(e)}"
            failed = True
        if not failed:
            try:
                # Handle the parsing step.
                JavaEntryPoint.parseQuery(generator, sql_str)
            except Exception as e:
                # Raise BodoError outside except to avoid stack trace
                func_text_or_error_msg = f"Failure encountered while parsing SQL Query. Error message: {error_to_string(e)}"
                failed = True
        if not failed:
            try:
                # Determine the write type
                write_type = JavaEntryPoint.getWriteType(generator, sql_str)

                # Get the row counts and NDV estimates for the tables:
                estimated_row_counts = []
                estimated_ndvs = []
                for i, table in enumerate(orig_bodo_types):
                    if isinstance(table, TablePathType):
                        row_count = table._statistics.get(
                            "row_count", bodo_sql_context_type.estimated_row_counts[i]
                        )
                        estimated_ndv = table._statistics.get("ndv", {})
                    else:
                        row_count = bodo_sql_context_type.estimated_row_counts[i]
                        estimated_ndv = {}
                    estimated_row_counts.append(row_count)
                    estimated_ndvs.append(estimated_ndv)

                # Update the schema with types.
                update_schema(
                    schema,
                    table_names,
                    df_types,
                    estimated_row_counts,
                    estimated_ndvs,
                    orig_bodo_types,
                    True,
                    write_type,
                )
                java_params_array = create_java_dynamic_parameter_type_list(
                    dynamic_param_values
                )
                named_params_dict = dict(zip(named_param_keys, named_param_values))
                java_named_params_map = create_java_named_parameter_type_map(
                    named_params_dict
                )
                code_plan_pair = JavaEntryPoint.getPandasAndPlanString(
                    generator,
                    sql_str,
                    True,
                    java_params_array,
                    java_named_params_map,
                )
                code = JavaEntryPoint.getCodeFromPair(code_plan_pair)
                plan = JavaEntryPoint.getPlanFromPair(code_plan_pair)
                # Convert to tuple of string tuples, to allow bcast to work
                globalsToLower = tuple(
                    [
                        (str(k), str(v))
                        for k, v in JavaEntryPoint.getLoweredGlobals(generator).items()
                    ]
                )
            except Exception as e:
                # Raise BodoError outside except to avoid stack trace
                func_text_or_error_msg = f"Failure in compiling or validating SQL Query. Error message: {error_to_string(e)}"
                failed = True
            if not failed:
                dynamic_param_names = [
                    DYNAMIC_PARAM_ARG_PREFIX + str(i)
                    for i in range(len(dynamic_param_values))
                ]
                named_param_names = [
                    NAMED_PARAM_ARG_PREFIX + x for x in named_param_keys
                ]
                args = ",".join(
                    ["bodo_sql_context"] + dynamic_param_names + named_param_names
                )
                func_text_or_error_msg = f"def impl({args}):\n"
                func_text_or_error_msg += f"{code}\n"

    failed = bcast_scalar(failed)
    func_text_or_error_msg = bcast_scalar(func_text_or_error_msg)
    if failed:
        raise bodo.utils.typing.BodoError(func_text_or_error_msg)
    plan = comm.bcast(plan)
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
                "time": time,
                "pd": pd,
                "datetime": datetime,
                "bif": bodo.ir.filter,
                "np": np,
            },
            locs,
        )
        outGlobalsDict[varname] = locs["value"]
    return func_text_or_error_msg, outGlobalsDict, plan


def _gen_sql_plan_pd_func_and_glbls_for_query(
    bodo_sql_context_type,
    sql_str,
    dynamic_param_values,
    named_param_keys,
    named_param_values,
):
    """Generate a Pandas function for query given the data type of SQL context.
    Used in Bodo typing pass to handle BodoSQLContext.sql() calls
    """
    import bodosql

    func_text, glblsToLower, sql_plan = _gen_sql_plan_pd_func_text_and_lowered_globals(
        bodo_sql_context_type,
        sql_str,
        dynamic_param_values,
        named_param_keys,
        named_param_values,
        False,  # Don't hide credentials because we need to execute this code.
    )

    glbls = {
        "pd": pd,
        "np": np,
        "bodo": bodo,
        "re": re,
        "bodosql": bodosql,
        "time": time,
        "datetime": datetime,
        "bif": bodo.ir.filter,
    }

    glbls.update(glblsToLower)

    loc_vars = {}
    exec(
        func_text,
        glbls,
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl, glblsToLower, sql_plan


@overload_method(BodoSQLContextType, "sql", inline="always", no_unliteral=True)
def overload_sql(bodo_sql_context, sql_str, params_dict=None, dynamic_params_list=None):
    """BodoSQLContextType.sql() should be handled in bodo typing pass since the
    generated code cannot be handled in regular overloads
    (requires Bodo's untyped pass and typing pass)
    """
    bodo.utils.typing.raise_bodo_error("Invalid BodoSQLContext.sql() call")


def _gen_pd_func_str_for_query(
    bodo_sql_context_type,
    sql_str,
    dynamic_param_values,
    named_param_keys,
    named_param_values,
):
    """Generate a function that returns the string of code that would be generated
    for the query given the data type of SQL context.
    Used in Bodo's typing pass to handle BodoSQLContext.convert_to_pandas() calls
    """

    # Don't need globals or the plan here, just need the func_text
    (
        returned_func_text,
        globalsToLower,
        _,
    ) = _gen_sql_plan_pd_func_text_and_lowered_globals(
        bodo_sql_context_type,
        sql_str,
        dynamic_param_values,
        named_param_keys,
        named_param_values,
        True,  # Hide credentials because we want to inspect the code, not run it.
    )

    # In this case, since the func_text is not going to be executed, we just
    # replace the lowered globals with the original values
    for k, v in globalsToLower.items():
        returned_func_text = returned_func_text.replace(str(k), str(v))

    executed_func_text = "def impl(bodo_sql_context):\n"
    # This doesn't work if we have triple quotes within the generated code
    # This currently isn't an issue, but I'm making note of it just in case
    executed_func_text += f'  return """{returned_func_text}"""'
    loc_vars = {}
    imports = {}
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
def overload_convert_to_pandas(
    bodo_sql_context, sql_str, params_dict=None, dynamic_params_list=None
):
    """BodoSQLContextType.convert_to_pandas() should be handled in bodo typing pass since the
    generated code cannot be handled in regular overloads
    (requires Bodo's untyped pass and typing pass)
    """
    bodo.utils.typing.raise_bodo_error(
        "Invalid BodoSQLContext.convert_to_pandas() call"
    )


# Scalar dtypes for supported Bodo Arrays
_numba_to_sql_column_type_map = {
    bodo.types.null_dtype: SqlTypeEnum.Null.value,
    types.int8: SqlTypeEnum.Int8.value,
    types.uint8: SqlTypeEnum.UInt8.value,
    types.int16: SqlTypeEnum.Int16.value,
    types.uint16: SqlTypeEnum.UInt16.value,
    types.int32: SqlTypeEnum.Int32.value,
    types.uint32: SqlTypeEnum.UInt32.value,
    types.int64: SqlTypeEnum.Int64.value,
    types.uint64: SqlTypeEnum.UInt64.value,
    types.float32: SqlTypeEnum.Float32.value,
    types.float64: SqlTypeEnum.Float64.value,
    types.NPDatetime("ns"): SqlTypeEnum.Timestamp_Ntz.value,
    types.NPTimedelta("ns"): SqlTypeEnum.Timedelta.value,
    bodo.types.pd_timedelta_type: SqlTypeEnum.Timedelta.value,
    types.bool_: SqlTypeEnum.Bool.value,
    bodo.types.string_type: SqlTypeEnum.String.value,
    bodo.types.bytes_type: SqlTypeEnum.Binary.value,
    # Note date doesn't have native support yet, but the code to
    # cast to datetime64 is handled in the Java code.
    bodo.types.datetime_date_type: SqlTypeEnum.Date.value,
    bodo.types.timestamptz_type: SqlTypeEnum.Timestamp_Tz.value,
}

# Scalar dtypes for supported parameters
_numba_to_sql_param_type_map = {
    types.none: SqlTypeEnum.Null.value,
    types.int8: SqlTypeEnum.Int8.value,
    types.uint8: SqlTypeEnum.UInt8.value,
    types.int16: SqlTypeEnum.Int16.value,
    types.uint16: SqlTypeEnum.UInt16.value,
    types.int32: SqlTypeEnum.Int32.value,
    types.uint32: SqlTypeEnum.UInt32.value,
    types.int64: SqlTypeEnum.Int64.value,
    types.uint64: SqlTypeEnum.UInt64.value,
    types.float32: SqlTypeEnum.Float32.value,
    types.float64: SqlTypeEnum.Float64.value,
    types.bool_: SqlTypeEnum.Bool.value,
    bodo.types.string_type: SqlTypeEnum.String.value,
    # Scalar datetime and timedelta are assumed
    # to be scalar Pandas Timestamp/Timedelta
    bodo.types.pd_timestamp_tz_naive_type: SqlTypeEnum.Timestamp_Ntz.value,
    bodo.types.timestamptz_type: SqlTypeEnum.Timestamp_Tz.value,
    # TODO: Support Date and Binary parameters [https://bodo.atlassian.net/browse/BE-3542]
}


def construct_tz_aware_array_type(typ, nullable):
    """Construct a BodoSQL data type for a tz-aware timestamp array

    Args:
        typ (types.Type): A tz-aware Bodo type
        nullable (bool): Is the column Nullable

    Returns:
        JavaObject: The Java Object for the BodoSQL column type data info.
    """
    # Timestamps only support precision 9 right now.
    precision = 9
    if typ.tz is None:
        # TZ = None is a timezone naive timestamp
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Timestamp_Ntz.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable, precision)
    else:
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Timestamp_Ltz.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable, precision)


def construct_time_array_type(
    typ: bodo.types.TimeArrayType | bodo.types.TimeType, nullable: bool
):
    """Construct a BodoSQL data type for a time array.

    Args:
        typ (Union[bodo.types.TimeArrayType, bodo.types.TimeType]): A time Bodo type
        nullable (bool): Is the column Nullable

    Returns:
        JavaObject: The Java Object for the BodoSQL column type data info.
    """
    type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
        SqlTypeEnum.Time.value
    )
    return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable, typ.precision)


def construct_array_item_array_type(arr_type):
    """Construct a BodoSQL data type for an array item array
    value.

    Args:
        typ (bodo.types.ArrayItemArrayType): A ArrayItemArray type
        col_name (str): Column name

    Returns:
        JavaObject: The Java Object for the BodoSQL column type data info.
    """
    child = get_sql_data_type(arr_type.dtype)
    type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
        SqlTypeEnum.Array.value
    )
    return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, True, child)


def construct_json_array_type(arr_type):
    """Construct a BodoSQL data type for a JSON array
    value.

    Args:
        typ (bodo.types.StructArrayType or bodo.types.MapArrayType): A StructArray or MapArray type
        col_name (str): Column name

    Returns:
        JavaObject: The Java Object for the BodoSQL column type data info.
    """
    if isinstance(arr_type, bodo.types.StructArrayType):
        # TODO: FIXME. We don't support full structs of types yet.
        # As a placeholder we will just match Snowflake.
        key_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.String.value
        )
        key = JavaEntryPoint.buildColumnDataTypeInfo(key_enum, True)
        value_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Variant.value
        )
        value = JavaEntryPoint.buildColumnDataTypeInfo(value_enum, True)
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Json_Object.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, True, key, value)
    else:
        # TODO: Add map scalar support
        key = get_sql_data_type(arr_type.key_arr_type)
        value = get_sql_data_type(arr_type.value_arr_type)
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Json_Object.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, True, key, value)


def get_sql_column_type_jit(arr_type, col_name):
    data_type = get_sql_data_type(arr_type)
    return JavaEntryPoint.buildBodoSQLColumnImpl(col_name, data_type)


def get_sql_data_type(arr_type):
    """get SQL type for a given array type."""
    warning_msg = f"Encountered type {arr_type} which is not supported in BodoSQL. BodoSQL will attempt to optimize the query to remove this column, but this can lead to errors in compilation. Please refer to the supported types: https://docs.bodo.ai/latest/source/BodoSQL.html#supported-data-types"
    # We currently treat NaT as nullable in BodoSQL, so for any array that has timestamp elements
    # type, we treat it as nullable.
    dtype_has_nullable = arr_type.dtype in (
        bodo.types.datetime64ns,
        bodo.types.timedelta64ns,
    )
    nullable = dtype_has_nullable or bodo.utils.typing.is_nullable_type(arr_type)
    if isinstance(arr_type, bodo.types.DatetimeArrayType):
        # Timezone-aware Timestamp columns have their own special handling.
        return construct_tz_aware_array_type(arr_type, nullable)
    elif arr_type == bodo.types.timestamptz_array_type:
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Timestamp_Tz.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable)
    elif isinstance(arr_type, bodo.types.TimeArrayType):
        # Time array types have their own special handling for precision
        return construct_time_array_type(arr_type, nullable)
    elif isinstance(arr_type, bodo.types.DecimalArrayType):
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Decimal.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(
            type_enum, nullable, arr_type.precision, arr_type.scale
        )
    elif isinstance(arr_type, bodo.types.ArrayItemArrayType):
        return construct_array_item_array_type(arr_type)
    elif isinstance(arr_type, (bodo.types.StructArrayType, bodo.types.MapArrayType)):
        return construct_json_array_type(arr_type)
    elif arr_type.dtype in _numba_to_sql_column_type_map:
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            _numba_to_sql_column_type_map[arr_type.dtype]
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable)
    elif isinstance(arr_type.dtype, bodo.types.PDCategoricalDtype):
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Categorical.value
        )
        child = get_sql_data_type(dtype_to_array_type(arr_type.dtype.elem_type, True))
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable, child)
    else:
        # The type is unsupported we raise a warning indicating this is a possible
        # error but we generate a dummy type because we may be able to support it
        # if its optimized out.
        warnings.warn(BodoSQLWarning(warning_msg))
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Unsupported.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable)


def get_sql_param_column_type_info(param_type: types.Type):
    """Get the SQL type information for a given Dynamic
    parameter type.

    Args:
        param_type (types.Type): The bodo type to lower as a parameter.
    Return:
        JavaObject: The ColumnDataTypeInfo for the parameter type.
    """
    unliteral_type = types.unliteral(param_type)
    # The named parameters are always scalars. We don't support
    # Optional types or None types yet. As a result this is always
    # non-null.
    nullable = False
    if (
        isinstance(unliteral_type, bodo.types.PandasTimestampType)
        and unliteral_type.tz != None
    ):
        return construct_tz_aware_array_type(param_type, nullable)
    elif isinstance(unliteral_type, bodo.types.TimeType):
        # Time array types have their own special handling for precision
        return construct_time_array_type(param_type, nullable)
    elif isinstance(unliteral_type, bodo.types.Decimal128Type):
        # Decimal types need handling for precision and scale.
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            SqlTypeEnum.Decimal.value
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(
            type_enum, nullable, unliteral_type.precision, unliteral_type.scale
        )
    elif unliteral_type in _numba_to_sql_param_type_map:
        type_enum = JavaEntryPoint.buildBodoSQLColumnDataTypeFromTypeId(
            _numba_to_sql_param_type_map[unliteral_type]
        )
        return JavaEntryPoint.buildColumnDataTypeInfo(type_enum, nullable)
    raise TypeError(
        f"Dynamic Parameter with type {param_type} not supported in BodoSQL. Please cast your data to a supported type. https://docs.bodo.ai/latest/source/BodoSQL.html#supported-data-types"
    )


def compute_df_types(df_list, is_bodo_type):
    """Given a list of Bodo types or Python objects,
    determines the DataFrame type for each object. This
    is used by both Python and JIT, where Python converts to
    Bodo types via the is_bodo_type argument. This function
    converts any TablePathType to the actual DataFrame type,
    which must be done in parallel.

    Args:
        df_list (List[types.Type | pd.DataFrame | bodosql.TablePath]):
            List of table either from Python or JIT.
        is_bodo_type (bool): Is this being called from JIT? If so we
            don't need to get the type of each member of df_list

    Raises:
        BodoError: If a TablePathType is passed with invalid
            values we raise an exception.

    Returns:
        Tuple(orig_bodo_types, df_types): Returns the Bodo types and
            the bodo.types.DataFrameType for each table. The original bodo
            types are kept to determine when code needs to be generated
            for TablePathType
    """

    orig_bodo_types = []
    df_types = []
    for df_val in df_list:
        if is_bodo_type:
            typ = df_val
        else:
            typ = bodo.typeof(df_val)
        orig_bodo_types.append(typ)

        if isinstance(typ, TablePathType):
            table_info = typ
            file_type = table_info._file_type
            file_path = table_info._file_path
            if file_type == "pq":
                # Extract the parquet information using Bodo
                type_info = bodo.io.parquet_pio.parquet_file_schema(file_path, None)
                # Future proof against additional return values that are unused
                # by BodoSQL by returning a tuple.
                col_names = type_info[0]
                col_types = type_info[1]
                index_cols = type_info[2]

                # If index_cols is empty or a single dict, then the index is a RangeIndex
                if (
                    len(index_cols) == 0
                    or len(index_cols) == 1
                    and isinstance(index_cols[0], dict)
                ):
                    index_col = index_cols[0] if len(index_cols) == 1 else None
                    if isinstance(index_col, dict) and index_col["name"] is not None:
                        index_col_name = types.StringLiteral(index_col["name"])
                    else:
                        index_col_name = None
                    index_typ = bodo.types.RangeIndexType(index_col_name)

                # Otherwise the index is a specific set of columns
                # Multiple for MultiIndex, single for single index
                else:
                    index_col_names = []
                    index_col_types = []
                    for index_col in index_cols:
                        # if the index_col is __index_level_0_, it means it has no name.
                        # Thus we do not write the name instead of writing '__index_level_0_' as the name
                        if "__index_level_" in index_col:
                            index_name = types.none
                        else:
                            index_name = types.StringLiteral(index_col)
                        # Convert the column type to an index type
                        index_loc = col_names.index(index_col)

                        index_col_types.append(col_types[index_loc])
                        index_col_names.append(index_name)

                        # Remove the index from the DataFrame.
                        col_names.pop(index_loc)
                        col_types.pop(index_loc)

                    if len(index_col_names) == 1:
                        index_elem_dtype = index_col_types[0].dtype
                        index_typ = bodo.utils.typing.index_typ_from_dtype_name_arr(
                            index_elem_dtype, index_col_names[0], index_col_types[0]
                        )
                    else:
                        bodo.hiframes.pd_multi_index_ext.MultiIndexType(
                            tuple(index_col_types),
                            tuple(index_col_names),
                        )

            elif file_type == "sql":
                const_conn_str = table_info._conn_str
                db_type, _ = parse_dbtype(const_conn_str)
                if db_type == "iceberg":
                    db_schema = table_info._db_schema
                    iceberg_table_name = table_info._file_path
                    # table_name = table_info.
                    type_info = bodo.transforms.untyped_pass
                    # schema = table_info._schema
                    (
                        col_names,
                        col_types,
                        _pyarrow_table_schema,
                    ) = bodo.io.iceberg.read_compilation.get_iceberg_orig_schema(
                        const_conn_str,
                        f"{db_schema}.{iceberg_table_name}",
                    )
                else:
                    type_info = (
                        bodo.transforms.untyped_pass._get_sql_types_arr_colnames(
                            f"{file_path}",
                            const_conn_str,
                            # _bodo_read_as_dict
                            None,
                            ir.Var(None, "dummy_var", ir.Loc("dummy_loc", -1)),
                            ir.Loc("dummy_loc", -1),
                            # is_table_input
                            True,
                            False,
                            # downcast_decimal_to_double
                            False,
                            convert_snowflake_column_names=False,
                        )
                    )
                    # Future proof against additional return values that are unused
                    # by BodoSQL by returning a tuple.
                    col_names = type_info[1]
                    col_types = type_info[3]

                # Generate the index type. We don't support an index column,
                # so this is always a RangeIndex.
                index_typ = bodo.types.RangeIndexType(None)
            else:
                raise BodoError(
                    "Internal error, 'compute_df_types' found a TablePath with an invalid file type"
                )

            # Generate the DataFrame type
            df_type = bodo.types.DataFrameType(
                tuple(col_types),
                index_typ,
                tuple(col_names),
            )
        else:
            df_type = typ
        df_types.append(df_type)
    return orig_bodo_types, df_types
