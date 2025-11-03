from __future__ import annotations

import datetime
import os
import re
import time
import traceback
import warnings
from typing import Any

import numba
import numpy as np
import pandas as pd
import pyarrow as pa

import bodo
import bodosql
from bodo.mpi4py import MPI
from bodosql.bodosql_types.database_catalog import DatabaseCatalog
from bodosql.bodosql_types.table_path import TablePath
from bodosql.imported_java_classes import (
    JavaEntryPoint,
)
from bodosql.plan_conversion import java_plan_to_python_plan
from bodosql.utils import BodoSQLWarning, error_to_string

# Prefix to add to table argument names when passed to JIT to avoid variable name conflicts
TABLE_ARG_PREFIX = "_ARG_"
# Prefix to add to bind variable argument names when passed to JIT to avoid variable name conflicts
DYNAMIC_PARAM_ARG_PREFIX = "_DYNAMIC_PARAM_"
# Prefix to add to named parameter argument names when passed to JIT to avoid variable name conflicts
NAMED_PARAM_ARG_PREFIX = "_NAMED_PARAM_"


class _CPPBackendExecutionFailed:
    """Sentinel class to indicate C++ backend execution failed and we should fall back to JIT"""

    pass


CPP_BACKEND_EXECUTION_FAILED = _CPPBackendExecutionFailed()


def _get_estimated_row_count(table: pd.DataFrame | TablePath) -> int | None:
    if isinstance(table, pd.DataFrame):
        # TODO: Handle distributed inputs.
        # Generate lengths if known.
        return len(table)
    elif isinstance(table, TablePath):
        return table.estimated_row_count
    else:
        # Pass None for unknown lengths.
        # TODO: Support other inputs types
        return None


def _get_estimated_ndv(table: pd.DataFrame | TablePath) -> dict[str, int]:
    if isinstance(table, TablePath):
        return table._statistics.get("ndv", {})
    else:
        return {}


class BodoSQLContext:
    def __init__(self, tables=None, catalog=None, default_tz=None):
        # We only need to initialize the tables values on all ranks, since that is needed for
        # creating the JIT function on all ranks for bc.sql calls. We also initialize df_types on all ranks,
        # for consistency. All the other attributes
        # are only used for generating the func text, which is only done on rank 0.
        if tables is None:
            tables = {}

        self.tables = tables
        self.default_tz = default_tz
        # Check types
        if any(not isinstance(key, str) for key in self.tables.keys()):
            raise ValueError("BodoSQLContext(): 'table' keys must be strings")
        if any(
            not isinstance(value, (pd.DataFrame, TablePath))
            for value in self.tables.values()
        ):
            raise ValueError(
                "BodoSQLContext(): 'table' values must be DataFrames or TablePaths"
            )

        if not (catalog is None or isinstance(catalog, DatabaseCatalog)):
            raise ValueError(
                "BodoSQLContext(): 'catalog' must be a bodosql.DatabaseCatalog if provided"
            )
        self.catalog = catalog

        # This except block can run in the case that our iceberg connector raises an error
        failed = False
        msg = ""
        try:
            # Convert to a dictionary mapping name -> type. For consistency
            # we first unpack the dictionary.
            names = []
            dfs = []
            estimated_row_counts = []
            estimated_ndvs = []
            for k, v in tables.items():
                names.append(k)
                dfs.append(v)
                estimated_row_counts.append(_get_estimated_row_count(v))
                estimated_ndvs.append(_get_estimated_ndv(v))
            schema = initialize_schema()
            self.schema = schema
            self.dfs = dfs
            self.names = names
            self.estimated_row_counts = estimated_row_counts
            self.estimated_ndvs = estimated_ndvs
        except Exception as e:
            failed = True
            msg = error_to_string(e)

        comm = MPI.COMM_WORLD
        failed = comm.bcast(failed)
        msg = comm.bcast(msg)
        if failed:
            raise ValueError(msg)

    def __getstate__(self) -> object:
        """
        Returns a state object used during pickling.
        """
        # 'schema' is a Java Object which cannot be pickled, so we
        # remove it from the state. We will re-initialize it during
        # unpickling (see __setstate__).
        dict_cp = self.__dict__.copy()
        dict_cp.pop("schema")
        return dict_cp

    def __setstate__(self, state):
        """
        Inverse of __getstate__ where we modify this
        object using the provided state.
        """
        # Set the state and initialize the Java objects from scratch.
        self.__dict__ = state
        self.schema = initialize_schema()

    def validate_query_compiles(self, sql, params_dict=None, dynamic_params_list=None):
        """
        Verifies BodoSQL can fully compile the query in Bodo.
        """
        try:
            t1 = time.time()
            self._compile(sql, params_dict, dynamic_params_list)
            compile_time = time.time() - t1
            compiles_flag = True
            error_message = "No error"
        except Exception as e:
            stack_trace = traceback.format_exc()
            compile_time = time.time() - t1
            compiles_flag = False
            error_message = repr(e)
            if os.environ.get("NUMBA_DEVELOPER_MODE", False):
                error_message = error_message + "\n" + stack_trace

        return compiles_flag, compile_time, error_message

    def _compile(self, sql, params_dict=None, dynamic_params_list=None):
        """compiles the query in Bodo."""
        import bodosql

        if params_dict is None:
            params_dict = {}

        dynamic_params_list = _ensure_dynamic_params_list(dynamic_params_list)

        generator = self._create_planner_and_parse_query(
            sql,
            False,  # We need to execute the code so don't hide credentials.
        )
        if bodo.get_rank() == 0:
            is_ddl = JavaEntryPoint.isDDLProcessedQuery(generator)
        else:
            is_ddl = False
        comm = MPI.COMM_WORLD
        is_ddl = comm.bcast(is_ddl)
        if is_ddl:
            warning_msg = "Encountered a DDL query. These queries are executed directly by bc.sql() so this wont't properly test compilation."
            warnings.warn(BodoSQLWarning(warning_msg))
        func_text, lowered_globals = self._convert_to_pandas(
            sql,
            dynamic_params_list,
            params_dict,
            generator,
            is_ddl,
        )

        glbls = {
            "np": np,
            "pd": pd,
            "bodosql": bodosql,
            "re": re,
            "bodo": bodo,
            "ColNamesMetaType": bodo.utils.typing.ColNamesMetaType,
            "MetaType": bodo.utils.typing.MetaType,
            "numba": numba,
            "time": time,
            "datetime": datetime,
            "bif": bodo.ir.filter,
        }

        glbls.update(lowered_globals)
        return self._functext_compile(
            func_text, dynamic_params_list, params_dict, glbls
        )

    def _functext_compile(self, func_text, dynamic_params_list, params_dict, glbls):
        """
        Helper function for _compile, that compiles the function text.
        This is mostly separated out for testing purposes.
        """

        arg_types = []
        for table_arg in self.tables.values():
            arg_types.append(bodo.typeof(table_arg))
        for dynamic_param_arg in dynamic_params_list:
            arg_types.append(bodo.typeof(dynamic_param_arg))
        for param_arg in params_dict.values():
            arg_types.append(bodo.typeof(param_arg))

        sig = tuple(arg_types)

        loc_vars = {}
        exec(
            func_text,
            glbls,
            loc_vars,
        )
        impl = loc_vars["bodosql_impl"]

        dispatcher = bodo.jit(sig)(impl)
        return dispatcher

    def validate_query(self, sql):
        """
        Verifies BodoSQL can compute query,
        but does not actually compile the query in Bodo.
        """
        try:
            self.convert_to_pandas(sql)
            executable_flag = True
        except Exception:
            executable_flag = False

        return executable_flag

    def convert_to_pandas(
        self, sql, params_dict=None, dynamic_params_list=None, hide_credentials=True
    ):
        """converts SQL code to Pandas"""
        if params_dict is None:
            params_dict = {}

        dynamic_params_list = _ensure_dynamic_params_list(dynamic_params_list)

        generator = self._create_planner_and_parse_query(
            sql,
            hide_credentials,
        )
        if bodo.get_rank() == 0:
            is_ddl = JavaEntryPoint.isDDLProcessedQuery(generator)
        else:
            is_ddl = False
        comm = MPI.COMM_WORLD
        is_ddl = comm.bcast(is_ddl)
        if is_ddl:
            warning_msg = "Encountered a DDL query. These queries are executed directly by bc.sql() so this wont't properly represent generated code."
            warnings.warn(BodoSQLWarning(warning_msg))
        pd_code, lowered_globals = self._convert_to_pandas(
            sql,
            dynamic_params_list,
            params_dict,
            generator,
            is_ddl,
        )
        # add the imports so someone can directly run the code.
        imports = [
            "import numpy as np",
            "import pandas as pd",
            "import time",
            "import datetime",
            "import numba",
            "import bodo",
            "import bodosql",
            "from bodo.utils.typing import ColNamesMetaType",
            "from bodo.utils.typing import MetaType",
            "import bodo.ir.filter as bif",
        ]
        added_globals = []
        # Add a decorator so someone can directly run the code.
        decorator = "@bodo.jit\n"
        # Add the global variable definitions at the beginning of the fn,
        # for better readability
        for varname, glbl in lowered_globals.items():
            added_globals.append(varname + " = " + repr(glbl))

        return (
            "\n".join(imports)
            + "\n"
            + "\n".join(added_globals)
            + "\n"
            + decorator
            + pd_code
        )

    def _create_planner_and_parse_query(self, sql: str, hide_credentials: bool):
        from bodo.mpi4py import MPI
        from bodosql.context_ext import compute_df_types, update_schema

        comm = MPI.COMM_WORLD

        plan_generator = None
        error_message = None
        if bodo.get_rank() == 0:
            plan_generator = self._create_generator(hide_credentials)
            try:
                if sql.strip() == "":
                    bodo.utils.typing.raise_bodo_error(
                        "BodoSQLContext passed empty query string"
                    )
                JavaEntryPoint.parseQuery(plan_generator, sql)
                # Write type is used for the current Merge Into code path decisions.
                # This should be removed when we revisit Merge Into
                write_type = JavaEntryPoint.getWriteType(plan_generator, sql)
                orig_bodo_types, df_types = compute_df_types(self.dfs, False)
                update_schema(
                    self.schema,
                    self.names,
                    df_types,
                    self.estimated_row_counts,
                    self.estimated_ndvs,
                    orig_bodo_types,
                    False,
                    write_type,
                )
            except Exception as e:
                error_message = error_to_string(e)

        error_message = comm.bcast(error_message)
        if error_message is not None:
            raise ValueError(
                f"Unable to parse SQL Query. Error message:\n{error_message}"
            )
        return plan_generator

    def _convert_to_pandas(
        self,
        sql: str,
        dynamic_params_list: list[Any],
        named_params_dict: dict[str, Any],
        generator,
        is_ddl: bool,
    ) -> tuple[str, dict[str, Any]]:
        """Generate the func_text for the Python code generated for the given SQL query.
        This is always computed entirely on rank 0 to avoid parallelism errors.

        Args:
            sql (str): The SQL query to process.
            dynamic_params_list (List[Any]): The list of dynamic parameters to lower.
            named_params_dict (Dict[str, Any]): The named parameters to lower.
            generator (RelationalAlgebraGenerator Java Object): The relational algebra generator
                used to generate the code.
            is_ddl (bool): Is this a DDL query?
        Raises:
            BodoError: If the SQL query cannot be processed.

        Returns:
            Tuple[str, Dict[str, Any]]: The generated code and the lowered global variables.
        """
        from bodo.mpi4py import MPI
        from bodo.utils.typing import BodoError

        comm = MPI.COMM_WORLD
        func_text_or_err_msg = ""
        failed = False
        globalsToLower = ()
        if bodo.get_rank() == 0:
            # This try block should never run under normal circumstances,
            # but it's nice to have for debugging purposes so things don't hang
            # if we make any changes that could lead to a runtime error.
            try:
                # Generate the code
                pd_code, globalsToLower = self._get_pandas_code(
                    sql, generator, dynamic_params_list, named_params_dict
                )
                # Convert to tuple of string tuples, to allow bcast to work
                globalsToLower = tuple(
                    [(str(k), str(v)) for k, v in globalsToLower.items()]
                )
                # Hard code the context name for DDL execution. This is used
                # for compilation testing and JIT code generation.
                context_names = ["bodo_sql_context"] if is_ddl else []
                table_names = [TABLE_ARG_PREFIX + x for x in self.tables.keys()]
                dynamic_param_names = [
                    DYNAMIC_PARAM_ARG_PREFIX + str(i)
                    for i in range(len(dynamic_params_list))
                ]
                named_param_names = [
                    NAMED_PARAM_ARG_PREFIX + x for x in named_params_dict.keys()
                ]
                args = ", ".join(
                    context_names
                    + table_names
                    + dynamic_param_names
                    + named_param_names
                )
                func_text_or_err_msg += f"def bodosql_impl({args}):\n"
                func_text_or_err_msg += f"{pd_code}\n"
            except Exception as e:
                failed = True
                func_text_or_err_msg = error_to_string(e)

        comm = MPI.COMM_WORLD
        failed = comm.bcast(failed)
        func_text_or_err_msg = comm.bcast(func_text_or_err_msg)
        if failed:
            raise BodoError(func_text_or_err_msg)

        globalsToLower = comm.bcast(globalsToLower)
        globalsDict = {}
        # convert the global map list of tuples of string varname and string value, to a map of string varname -> python value.
        for varname, str_value in globalsToLower:
            locs = {}
            exec(
                f"value = {str_value}",
                {
                    "ColNamesMetaType": bodo.utils.typing.ColNamesMetaType,
                    "MetaType": bodo.utils.typing.MetaType,
                    "bodo": bodo,
                    "numba": numba,
                    "time": time,
                    "pd": pd,
                    "datetime": datetime,
                    "bif": bodo.ir.filter,
                    "np": np,
                },
                locs,
            )
            globalsDict[varname] = locs["value"]
        return func_text_or_err_msg, globalsDict

    def sql(self, sql, params_dict=None, dynamic_params_list=None, **jit_options):
        import bodosql
        from bodo.spawn.spawner import SpawnDispatcher

        if params_dict is None:
            params_dict = {}

        dynamic_params_list = _ensure_dynamic_params_list(dynamic_params_list)

        generator = self._create_planner_and_parse_query(
            sql,
            False,  # We need to execute the code so don't hide credentials.
        )
        if bodo.get_rank() == 0:
            is_ddl = JavaEntryPoint.isDDLProcessedQuery(generator)
        else:
            is_ddl = False
        comm = MPI.COMM_WORLD
        is_ddl = comm.bcast(is_ddl)
        if is_ddl:
            # Just execute DDL operations directly and return the DataFrame.
            return self.execute_ddl(sql, generator)
        elif (
            bodosql.use_cpp_backend
            and (
                output := self.execute_cpp_backend(
                    sql, generator, dynamic_params_list, params_dict
                )
            )
            is not CPP_BACKEND_EXECUTION_FAILED
        ):
            return output
        else:
            func_text, lowered_globals = self._convert_to_pandas(
                sql,
                dynamic_params_list,
                params_dict,
                generator,
                False,  # This path is never DDL.s
            )
            glbls = {
                "np": np,
                "pd": pd,
                "bodosql": bodosql,
                "re": re,
                "bodo": bodo,
                "ColNamesMetaType": bodo.utils.typing.ColNamesMetaType,
                "MetaType": bodo.utils.typing.MetaType,
                "numba": numba,
                "time": time,
                "datetime": datetime,
                "bif": bodo.ir.filter,
            }

            glbls.update(lowered_globals)
            loc_vars = {}
            impl = bodo.utils.utils.bodo_spawn_exec(
                func_text, glbls, loc_vars, __name__
            )

            # Add table argument name prefix to user provided distributed flags to match
            # stored names
            if "distributed" in jit_options and isinstance(
                jit_options["distributed"], (list, set)
            ):
                jit_options["distributed"] = [
                    TABLE_ARG_PREFIX + x for x in jit_options["distributed"]
                ]
            if "replicated" in jit_options and isinstance(
                jit_options["replicated"], (list, set)
            ):
                jit_options["replicated"] = [
                    TABLE_ARG_PREFIX + x for x in jit_options["replicated"]
                ]

            dispatcher = bodo.jit(impl, **jit_options)

            # Save BodoSQL globals in SpawnDispatcher to be handled in pickling
            # properly. Internal CASE implementation strings may use some globals that
            # are not visible to cloudpickle as used by the function. See:
            # test_json_fns.py::test_object_construct_keep_null[no_nested-no_null-with_case]
            if isinstance(dispatcher, SpawnDispatcher):
                # __builtins__ which is added to glbls by exec causes issues in
                # Jupyter/IPython pickling.
                glbls.pop("__builtins__", None)
                dispatcher.add_extra_globals(glbls)

            return dispatcher(
                *(
                    list(self.tables.values())
                    + dynamic_params_list
                    + list(params_dict.values())
                )
            )

    def generate_plan(
        self, sql, params_dict=None, dynamic_params_list=None, show_cost=False
    ) -> str:
        """
        Return the optimized plan for the SQL code as
        as a Python string.
        """
        from bodosql.context_ext import (
            create_java_dynamic_parameter_type_list,
            create_java_named_parameter_type_map,
        )

        if params_dict is None:
            params_dict = {}

        dynamic_params_list = _ensure_dynamic_params_list(dynamic_params_list)

        generator = self._create_planner_and_parse_query(sql, True)
        failed = False
        plan_or_err_msg = ""
        if bodo.get_rank() == 0:
            try:
                java_params_array = create_java_dynamic_parameter_type_list(
                    dynamic_params_list
                )
                java_named_params_map = create_java_named_parameter_type_map(
                    params_dict
                )
                plan_or_err_msg = str(
                    JavaEntryPoint.getOptimizedPlanString(
                        generator,
                        sql,
                        show_cost,
                        java_params_array,
                        java_named_params_map,
                    )
                )
            except Exception as e:
                failed = True
                plan_or_err_msg = error_to_string(e)

        comm = MPI.COMM_WORLD
        failed = comm.bcast(failed)
        plan_or_err_msg = comm.bcast(plan_or_err_msg)
        if failed:
            raise ValueError(plan_or_err_msg)
        return plan_or_err_msg

    def _get_pandas_code(
        self,
        sql: str,
        generator,
        dynamic_params_list: list[Any],
        named_params_dict: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Generate the Pandas code for the given SQL string.

        Args:
            sql (str): The SQL query text.
            generator (RelationalAlgebraGenerator Java Object): The relational algebra generator
                used to generate the code.

        Raises:
            bodo.utils.typing.BodoError: The SQL text is not supported.

        Returns:
            Tuple[str, Dict[str, Any]]: The generated code and the lowered global variables.
        """
        from bodosql.context_ext import (
            create_java_dynamic_parameter_type_list,
            create_java_named_parameter_type_map,
        )

        # Construct the relational algebra generator
        try:
            java_params_array = create_java_dynamic_parameter_type_list(
                dynamic_params_list
            )
            java_named_params_map = create_java_named_parameter_type_map(
                named_params_dict
            )
            pd_code = str(
                JavaEntryPoint.getPandasString(
                    generator, sql, java_params_array, java_named_params_map
                )
            )
            failed = False
        except Exception as e:
            message = error_to_string(e)
            failed = True
        if failed:
            # Raise BodoError outside except to avoid stack trace
            raise bodo.utils.typing.BodoError(
                f"Unable to compile SQL Query. Error message:\n{message}"
            )
        return pd_code, JavaEntryPoint.getLoweredGlobals(generator)

    def execute_cpp_backend(
        self,
        sql: str,
        generator,
        dynamic_params_list: list[Any],
        named_params_dict: dict[str, Any],
    ) -> pd.DataFrame | None | _CPPBackendExecutionFailed:
        """Execute the query using the C++ backend if possible.

        Args:
            sql (str): The SQL query text.
            generator (RelationalAlgebraGenerator Java Object): The relational algebra generator
                used to generate the plan.

        Returns:
            pd.DataFrame | None | _CPPBackendExecutionFailed: The result of the query execution or a failure indicator.
        """
        from bodosql.context_ext import (
            create_java_dynamic_parameter_type_list,
            create_java_named_parameter_type_map,
        )

        try:
            java_params_array = create_java_dynamic_parameter_type_list(
                dynamic_params_list
            )
            java_named_params_map = create_java_named_parameter_type_map(
                named_params_dict
            )
            java_plan = JavaEntryPoint.getOptimizedPlan(
                generator, sql, java_params_array, java_named_params_map
            )
            # Keeps track of join ids and their join filter key locations for join
            # filter translation during conversion to Python plan.
            self.join_filter_info = {}
            plan = java_plan_to_python_plan(self, java_plan)
            out = bodo.pandas.plan.execute_plan(plan, optimize=False)
        except Exception as e:
            message = error_to_string(e)
            if bodosql.verbose_cpp_backend:
                print(f"C++ backend execution failed with error:\n{message}")
            if bodosql.cpp_backend_no_fallback:
                raise RuntimeError(
                    f"C++ backend execution failed with error:\n{message}"
                ) from e
            out = CPP_BACKEND_EXECUTION_FAILED
        finally:
            self.join_filter_info = None

        return out

    def _create_generator(self, hide_credentials: bool):
        """Creates a RelationalAlgebraGenerator from the schema.

        Args:
            hide_credentials (bool): Should credentials be hidden for
                any generated code.

        Returns:
            RelationalAlgebraGenerator Java Object: The java object holding
                the relational algebra generator.
        """
        verbose_level = bodo.user_logging.get_verbose_level()
        tracing_level = bodo.tracing_level
        if self.catalog is not None:
            catalog_obj = self.catalog.get_java_object()
        else:
            catalog_obj = None
        return JavaEntryPoint.buildRelationalAlgebraGenerator(
            catalog_obj,
            self.schema,
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
            self.default_tz,
        )

    def add_or_replace_view(
        self, name: str, table: pd.DataFrame | TablePath
    ) -> BodoSQLContext:
        """Create a new BodoSQLContext that contains all of the old DataFrames and the
        new table being provided. If there is a DataFrame in the old BodoSQLContext with
        the same name, it is replaced by the new table in the new BodoSQLContext. Otherwise
        the new table is just added under the new name.

        Args:
            name (str): Name of the new table
            table (Union[pd.DataFrame,  TablePath]): New tables

        Returns:
            BodoSQLContext: A new BodoSQL context.

        Raises BodoError
        """
        if not isinstance(name, str):
            raise ValueError(
                "BodoSQLContext.add_or_replace_view(): 'name' must be a string"
            )
        if not isinstance(table, (pd.DataFrame, TablePath)):
            raise ValueError(
                "BodoSQLContext.add_or_replace_view(): 'table' must be a Pandas DataFrame or BodoSQL TablePath"
            )
        new_tables = self.tables.copy()
        new_tables[name] = table
        return BodoSQLContext(new_tables, self.catalog, self.default_tz)

    def remove_view(self, name: str):
        """Create a new BodoSQLContext by removing the table with the
        given name.

        Args:
            name (str): Name of the table to remove.

        Returns:
            BodoSQLContext: A new BodoSQL context.

        Raises BodoError
        """
        if not isinstance(name, str):
            raise ValueError(
                "BodoSQLContext.remove_view(): 'name' must be a constant string"
            )
        new_tables = self.tables.copy()
        if name not in new_tables:
            raise ValueError(
                "BodoSQLContext.remove_view(): 'name' must refer to a registered view"
            )
        del new_tables[name]
        return BodoSQLContext(new_tables, self.catalog, self.default_tz)

    def add_or_replace_catalog(self, catalog: DatabaseCatalog):
        """
        Creates a new BodoSQL context by replacing the previous catalog,
        if it exists, with the provided catalog.

        Args:
            catalog (DatabaseCatalog): DatabaseCatalog to add to the context.

        Returns:
            BodoSQLContext: A new BodoSQL context.

        Raises BodoError
        """
        if not isinstance(catalog, DatabaseCatalog):
            raise ValueError(
                "BodoSQLContext.add_or_replace_catalog(): 'catalog' must be a bodosql.DatabaseCatalog"
            )
        return BodoSQLContext(self.tables, catalog, self.default_tz)

    def remove_catalog(self):
        """
        Creates a new BodoSQL context by remove the previous catalog.

        Returns:
            BodoSQLContext: A new BodoSQL context.

        Raises BodoError
        """
        if self.catalog is None:
            raise ValueError(
                "BodoSQLContext.remove_catalog(): BodoSQLContext must have an existing catalog registered."
            )
        return BodoSQLContext(self.tables, self.default_tz, self.default_tz)

    def __eq__(self, bc: object) -> bool:
        if isinstance(bc, BodoSQLContext):
            # Since the dictionary can contain either
            # DataFrames or table paths, we must add separate
            # checks for both.
            curr_keys = set(self.tables.keys())
            bc_keys = set(bc.tables.keys())
            if curr_keys == bc_keys:
                for key in curr_keys:
                    if isinstance(self.tables[key], TablePath) and isinstance(
                        bc.tables[key], TablePath
                    ):
                        if not self.tables[key].equals(
                            bc.tables[key]
                        ):  # pragma: no cover
                            return False
                    elif isinstance(self.tables[key], pd.DataFrame) and isinstance(
                        bc.tables[key], pd.DataFrame
                    ):  # pragma: no cover
                        # DataFrames may not have exactly the same dtypes becasue of flags inside boxing (e.g. object -> string)
                        # As a result we determine equality using assert_frame_equals
                        try:
                            pd.testing.assert_frame_equal(
                                self.tables[key],
                                bc.tables[key],
                                check_dtype=False,
                                check_index_type=False,
                            )
                        except AssertionError:
                            return False
                    else:
                        return False
                return self.catalog == bc.catalog
        return False  # pragma: no cover

    def execute_ddl(self, sql: str, generator=None) -> pd.DataFrame:
        """API to directly execute DDL queries. This is used by the JIT
        path to execute DDL queries and can be used as a fast path when you
        statically know the query you want to execute is a DDL query to avoid the
        control flow/cleanup code.

        This will execute any DDL query on rank 0 and then broadcast the result
        to all ranks.

        Args:
            sql (str): The DDL query to execute.
            generator (Optional[RelationalAlgebraGenerator Java object]): The prepared planner
                information used for executing the query. If None we need to create
                the planner.

        Returns:
            pd.DataFrame: The result of the DDL query as a Pandas DataFrame.
        """
        from bodo.mpi4py import MPI

        comm = MPI.COMM_WORLD
        result = None
        error = None
        create_generator = comm.bcast(generator is None)
        if create_generator:
            # Prepare the relational algebra generator on rank 0.
            # The assumption is this code is called directly as the
            # external API so we need to parse the query.
            generator = self._create_planner_and_parse_query(
                sql,
                False,  # We need to execute the code so don't hide credentials.
            )

        if bodo.get_rank() == 0:
            try:
                ddl_result = JavaEntryPoint.executeDDL(generator, sql)
                # Convert the output to a DataFrame.
                column_names = list(
                    JavaEntryPoint.getDDLExecutionColumnNames(ddl_result)
                )
                column_types = [
                    _generate_ddl_column_type(t)
                    for t in JavaEntryPoint.getDDLExecutionColumnTypes(ddl_result)
                ]
                data = [
                    # Use astype to avoid issues with Java conversion.
                    pd.array(column, dtype=object).astype(column_types[i])
                    for i, column in enumerate(
                        JavaEntryPoint.getDDLColumnValues(ddl_result)
                    )
                ]
                df_dict = {column_names[i]: data[i] for i in range(len(column_names))}
                result = pd.DataFrame(
                    df_dict,
                )
            except Exception as e:
                error = error_to_string(e)
        result = comm.bcast(result)
        error = comm.bcast(error)
        # Throw the error on all ranks.
        if error is not None:
            raise ValueError(error)
        return result


def _generate_ddl_column_type(type_string: str) -> Any:
    """Convert a string representation of a Pandas column type
    passed from Java to a Python type.

    Args:
        type_string (str): A string for the expression you would
            execution in Python to get the type.

    Returns:
        Any: The actual type object.
    """
    glbls = {"pd": pd, "pa": pa}
    locs = {}
    exec(f"ddl_type = {type_string}", glbls, locs)
    return locs["ddl_type"]


def initialize_schema():
    """Create the BodoSQL Schema used to store all local DataFrames.

    Returns:
        Java LocalSchema: Java type for the BodoSQL schema.
    """
    # TODO(ehsan): create and store generator during bodo_sql_context initialization
    if bodo.get_rank() == 0:
        schema = JavaEntryPoint.buildLocalSchema("__BODOLOCAL__")
    else:
        schema = None
    return schema


def _ensure_dynamic_params_list(dynamic_params_list: Any) -> list:
    """Verify the supplied Dynamic params list is a supported type
    and converts the result to a list.

    Args:
        dynamic_params_list (Any): A representation of the dynamic params list.

    Returns:
        List: The dynamic params list converted to a list equivalent.
    """
    if dynamic_params_list is None:
        return []
    elif isinstance(dynamic_params_list, tuple):
        return list(dynamic_params_list)
    elif isinstance(dynamic_params_list, list):
        return dynamic_params_list
    else:
        # Only specify tuple in the error message because we may not be able
        # to support lists in JIT.
        raise ValueError(
            "dynamic_params_list must be a tuple of Python variables if provided"
        )
