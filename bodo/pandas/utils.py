import functools
import importlib
import inspect

import numba
import pandas as pd
import pyarrow as pa
from llvmlite import ir as lir
from numba.extending import intrinsic

import bodo
from bodo.libs.array import cpp_table_to_py_table, delete_table, table_type
from bodo.pandas.array_manager import LazyArrayManager, LazySingleArrayManager
from bodo.pandas.managers import LazyBlockManager, LazySingleBlockManager
from bodo.utils.typing import check_unsupported_args_fallback


def get_data_manager_pandas() -> str:
    """Get the value of mode.data_manager from pandas config.

    Returns:
        str: The value of the mode.data_manager option or 'block'
    """
    try:
        from pandas._config.config import _get_option

        return _get_option("mode.data_manager", silent=True)
    except ImportError:
        # _get_option and mode.data_manager are not supported in Pandas > 2.2.
        return "block"


def get_lazy_manager_class() -> type[LazyArrayManager | LazyBlockManager]:
    """Get the lazy manager class based on the pandas option mode.data_manager, suitable for DataFrame."""
    data_manager = get_data_manager_pandas()
    if data_manager == "block":
        return LazyBlockManager
    elif data_manager == "array":
        return LazyArrayManager
    raise Exception(
        f"Got unexpected value of pandas option mode.manager: {data_manager}"
    )


def get_lazy_single_manager_class() -> type[
    LazySingleArrayManager | LazySingleBlockManager
]:
    """Get the lazy manager class based on the pandas option mode.data_manager, suitable for Series."""
    data_manager = get_data_manager_pandas()
    if data_manager == "block":
        return LazySingleBlockManager
    elif data_manager == "array":
        return LazySingleArrayManager
    raise Exception(
        f"Got unexpected value of pandas option mode.manager: {data_manager}"
    )


@intrinsic
def cast_int64_to_table_ptr(typingctx, val):
    """Cast int64 value to C++ table pointer"""

    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], lir.IntType(8).as_pointer())

    return table_type(numba.core.types.int64), codegen


@numba.njit
def cpp_table_to_py(in_table, out_cols_arr, out_table_type):
    """Convert a C++ table pointer to a Python table.
    Args:
        in_table (int64): C++ table pointer
        out_cols_arr (array(int64)): Array of column indices to be extracted
        out_table_type (types.Type): Type of the output table
    """
    cpp_table = cast_int64_to_table_ptr(in_table)
    out_table = cpp_table_to_py_table(cpp_table, out_cols_arr, out_table_type, 0)
    delete_table(cpp_table)
    return out_table


def cpp_table_to_df(cpp_table, arrow_schema):
    """Convert a C++ table (table_info) to a pandas DataFrame."""

    import numpy as np

    from bodo.hiframes.table import TableType
    from bodo.io.helpers import pyarrow_type_to_numba

    out_cols_arr = np.array(range(len(arrow_schema)), dtype=np.int64)
    table_type = TableType(
        tuple([pyarrow_type_to_numba(field.type) for field in arrow_schema])
    )

    out_df = cpp_table_to_py(cpp_table, out_cols_arr, table_type).to_pandas()
    out_df.columns = [f.name for f in arrow_schema]
    # TODO: handle Indexes properly
    if "__index_level_0__" in out_df.columns:
        out_df = out_df.drop(columns=["__index_level_0__"])
    return out_df


def cpp_table_to_series(cpp_table, arrow_schema):
    """Convert a C++ table (table_info) to a pandas Series."""

    as_df = cpp_table_to_df(cpp_table, arrow_schema)
    assert len(arrow_schema) == 1
    return as_df[arrow_schema[0].name]


@functools.lru_cache
def get_dataframe_overloads():
    """Return a list of the functions supported on BodoDataFrame objects
    to some degree by bodo.jit.
    """
    from bodo.hiframes.pd_dataframe_ext import DataFrameType
    from bodo.numba_compat import get_method_overloads

    return get_method_overloads(DataFrameType)


@functools.lru_cache
def get_series_overloads():
    """Return a list of the functions supported on BodoSeries objects
    to some degree by bodo.jit.
    """
    from bodo.hiframes.pd_series_ext import SeriesType
    from bodo.numba_compat import get_method_overloads

    return get_method_overloads(SeriesType)


def get_overloads(cls_name):
    """Use the class name of the __class__ attr of self parameter
    to determine which of the above two functions to call to
    get supported overloads for the current data type.
    """
    if cls_name == "BodoDataFrame":
        return get_dataframe_overloads()
    elif cls_name == "BodoSeries":
        return get_series_overloads()
    else:
        assert False


def check_args_fallback(
    unsupported=None, supported=None, package_name="pandas", fn_str=None, module_name=""
):
    """Decorator to apply to dataframe or series member functions that handles
    argument checking, falling back to JIT compilation when it might work, and
    falling back to Pandas if necessary.

    Parameters:
        unsupported -
            1) Can be "all" which means that all the parameters that have
               a default value must have that default value.  In other
               words, we don't support anything but the default value.
            2) Can be "none" which means that we support all the parameters
               that have a default value and you can set them to any allowed
               value.
            3) Can be a list of parameter names for which they must have their
               default value.  All non-listed parameters that have a default
               value are allowed to take on any allowed value.
        supported - a list of parameter names for which they can have something
               other than their default value.  All non-listed parameters that
               have a default value are not allowed to take on anything other
               than their default value.
        package_name - see bodo.utils.typing.check_unsupported_args_fallback
        fn_str - see bodo.utils.typing.check_unsupported_args_fallback
        module_name - see bodo.utils.typing.check_unsupported_args_fallback
    """
    assert (unsupported is None) ^ (supported is None), (
        "Exactly one of unsupported and supported must be specified."
    )

    def decorator(func):
        def to_bodo(val):
            if isinstance(val, pd.DataFrame):
                return bodo.pandas.DataFrame(val)
            elif isinstance(val, pd.Series):
                return bodo.pandas.Series(val)
            else:
                assert False, f"Unexpected val type {type(val)}"

        # See if function is top-level or not by looking for a . in
        # the full name.
        toplevel = "." not in func.__qualname__
        if not bodo.dataframe_library_enabled:
            # Dataframe library not enabled so just call the Pandas super class version.
            if toplevel:
                py_pkg = importlib.import_module(package_name)

                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    # Call the same method in the base class.
                    return to_bodo(getattr(py_pkg, func.__name__)(*args, **kwargs))
            else:

                @functools.wraps(func)
                def wrapper(self, *args, **kwargs):
                    # Call the same method in the base class.
                    return to_bodo(
                        getattr(self.__class__.__bases__[0], func.__name__)(
                            self, *args, **kwargs
                        )
                    )
        else:
            signature = inspect.signature(func)
            if unsupported == "all":
                unsupported_args = {
                    idx: param
                    for idx, (name, param) in enumerate(signature.parameters.items())
                    if param.default is not inspect.Parameter.empty
                }
                unsupported_kwargs = {
                    name: param
                    for name, param in signature.parameters.items()
                    if param.default is not inspect.Parameter.empty
                }
            elif unsupported == "none":
                unsupported_args = {}
                unsupported_kwargs = {}
            else:
                if supported is not None:
                    inverted = True
                    flist = supported
                else:
                    flist = unsupported
                unsupported_args = {
                    idx: param
                    for idx, (name, param) in enumerate(signature.parameters.items())
                    if (param.default is not inspect.Parameter.empty)
                    and (inverted ^ (name in flist))
                }
                unsupported_kwargs = {
                    name: param
                    for name, param in signature.parameters.items()
                    if (param.default is not inspect.Parameter.empty)
                    and (inverted ^ (name in flist))
                }

            if toplevel:
                py_pkg = importlib.import_module(package_name)

                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    from bodo.pandas import BODO_PANDAS_FALLBACK

                    error = check_unsupported_args_fallback(
                        func.__qualname__,
                        unsupported_args,
                        unsupported_kwargs,
                        args,
                        kwargs,
                        package_name=package_name,
                        fn_str=fn_str,
                        module_name=module_name,
                        raise_on_error=(BODO_PANDAS_FALLBACK == 0),
                    )
                    if error:
                        # Can we do a top-level override check?

                        # Fallback to Python. Call the same method in the base class.
                        return to_bodo(getattr(py_pkg, func.__name__)(*args, **kwargs))
                    else:
                        result = func(*args, **kwargs)
                    return result
            else:

                @functools.wraps(func)
                def wrapper(self, *args, **kwargs):
                    from bodo.pandas import BODO_PANDAS_FALLBACK

                    error = check_unsupported_args_fallback(
                        func.__qualname__,
                        unsupported_args,
                        unsupported_kwargs,
                        args,
                        kwargs,
                        package_name=package_name,
                        fn_str=fn_str,
                        module_name=module_name,
                        raise_on_error=(BODO_PANDAS_FALLBACK == 0),
                    )
                    if error:
                        # The dataframe library must not support some specified option.
                        # Get overloaded functions for this dataframe/series in JIT mode.
                        overloads = get_overloads(self.__class__.__name__)
                        if func.__name__ in overloads:
                            # TO-DO: Generate a function and bodo JIT it to do this
                            # individual operation.  If the compile fails then fallthrough
                            # to the pure Python code below.  If the compile works then
                            # run the operation using the JITted function.
                            pass

                        # Fallback to Python. Call the same method in the base class.
                        return to_bodo(
                            getattr(self.__class__.__bases__[0], func.__name__)(
                                self, *args, **kwargs
                            )
                        )
                    else:
                        result = func(self, *args, **kwargs)
                    return result

        return wrapper

    return decorator


class LazyPlan:
    """Easiest mode to use DuckDB is to generate isolated queries and try to minimize
    node re-use issues due to the frequent use of unique_ptr.  This class should be
    used when constructing all plans and holds them lazily.  On demand, generate_duckdb
    can be used to convert to an isolated set of DuckDB objects for execution.
    """

    def __init__(self, plan_class, *args, **kwargs):
        self.plan_class = plan_class
        self.args = args
        self.kwargs = kwargs
        self.output_func = None  # filled in by wrap_plan

    def generate_duckdb(self, cache=None):
        from bodo.ext import plan_optimizer

        # Sometimes the same LazyPlan object is encountered twice during the same
        # query so  we use the cache dict to only convert it once.
        if cache is None:
            cache = {}
        # If previously converted then use the last result.
        if id(self) in cache:
            return cache[id(self)]

        def recursive_check(x):
            """Recursively convert LazyPlans but return other types unmodified."""
            if isinstance(x, LazyPlan):
                return x.generate_duckdb(cache=cache)
            else:
                return x

        # Convert any LazyPlan in the args or kwargs.
        args = [recursive_check(x) for x in self.args]
        kwargs = {k: recursive_check(v) for k, v in self.kwargs.items()}
        # Create real duckdb class.
        ret = getattr(plan_optimizer, self.plan_class)(*args, **kwargs)
        # Add to cache so we don't convert it again.
        cache[id(self)] = ret
        return ret


def execute_plan(plan: LazyPlan):
    """Execute a dataframe plan using Bodo's execution engine.

    Args:
        plan (LazyPlan): query plan to execute

    Returns:
        pd.DataFrame: output data
    """
    import bodo

    def _exec_plan(plan):
        import bodo
        from bodo.ext import plan_optimizer

        duckdb_plan = plan.generate_duckdb()

        # Print the plan before optimization
        if bodo.tracing_level >= 2 and bodo.libs.distributed_api.get_rank() == 0:
            pre_optimize_graphviz = duckdb_plan.toGraphviz()
            with open("pre_optimize" + str(id(plan)) + ".dot", "w") as f:
                print(pre_optimize_graphviz, file=f)

        optimized_plan = plan_optimizer.py_optimize_plan(duckdb_plan)

        # Print the plan after optimization
        if bodo.tracing_level >= 2 and bodo.libs.distributed_api.get_rank() == 0:
            post_optimize_graphviz = optimized_plan.toGraphviz()
            with open("post_optimize" + str(id(plan)) + ".dot", "w") as f:
                print(post_optimize_graphviz, file=f)
        return plan_optimizer.py_execute_plan(optimized_plan, plan.output_func)

    if bodo.dataframe_library_run_parallel:
        import bodo.spawn.spawner

        return bodo.spawn.spawner.submit_func_to_workers(_exec_plan, [], plan)

    return _exec_plan(plan)


@intrinsic
def cast_table_ptr_to_int64(typingctx, val):
    """Cast C++ table pointer to int64 (to pass to C++ later)"""

    def codegen(context, builder, signature, args):
        return builder.ptrtoint(args[0], lir.IntType(64))

    return numba.core.types.int64(table_type), codegen


def df_to_cpp_table(df):
    """Convert a pandas DataFrame to a C++ table pointer and Arrow schema object."""
    n_cols = len(df.columns)
    in_col_inds = bodo.utils.typing.MetaType(tuple(range(n_cols)))

    @numba.jit
    def impl_df_to_cpp_table(df):
        table = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df)
        cpp_table = bodo.libs.array.py_data_to_cpp_table(table, (), in_col_inds, n_cols)
        return cast_table_ptr_to_int64(cpp_table)

    cpp_table = impl_df_to_cpp_table(df)
    arrow_schema = pa.Schema.from_pandas(df)

    return cpp_table, arrow_schema


def run_apply_udf(cpp_table, arrow_schema, func):
    """Run a user-defined function (UDF) on a DataFrame created from C++ table and
    return the result as a C++ table and Arrow schema.
    """
    df = cpp_table_to_df(cpp_table, arrow_schema)
    out_df = pd.DataFrame({"OUT": df.apply(func, axis=1)})
    return df_to_cpp_table(out_df)


def _del_func(x):
    # Intentionally do nothing
    pass


def wrap_plan(schema, plan, res_id=None, nrows=None, index_data=None):
    """Create a BodoDataFrame or BodoSeries with the given
    schema and given plan node.
    """
    import pandas as pd

    from bodo.pandas.frame import BodoDataFrame
    from bodo.pandas.lazy_metadata import LazyMetadata
    from bodo.pandas.series import BodoSeries
    from bodo.pandas.utils import (
        LazyPlan,
        get_lazy_manager_class,
        get_lazy_single_manager_class,
    )

    assert isinstance(plan, LazyPlan), "wrap_plan: LazyPlan expected"

    if isinstance(schema, dict):
        schema = {
            col: pd.Series(dtype=col_type.dtype) for col, col_type in schema.items()
        }

    if nrows is None:
        # Fake non-zero rows.  nrows should be overwritten upon plan execution.
        nrows = 1

    if isinstance(schema, (dict, pd.DataFrame)):
        if isinstance(schema, dict):
            schema = pd.DataFrame(schema)
        metadata = LazyMetadata(
            "LazyPlan_" + str(plan.plan_class) if res_id is None else res_id,
            schema,
            nrows=nrows,
            index_data=index_data,
        )
        mgr = get_lazy_manager_class()
        new_df = BodoDataFrame.from_lazy_metadata(
            metadata, collect_func=mgr._collect, del_func=_del_func, plan=plan
        )
        plan.output_func = cpp_table_to_df
    elif isinstance(schema, pd.Series):
        metadata = LazyMetadata(
            "LazyPlan_" + str(plan.plan_class) if res_id is None else res_id,
            schema,
            nrows=nrows,
            index_data=index_data,
        )
        mgr = get_lazy_single_manager_class()
        new_df = BodoSeries.from_lazy_metadata(
            metadata, collect_func=mgr._collect, del_func=_del_func, plan=plan
        )
        plan.output_func = cpp_table_to_series
    else:
        assert False

    new_df.plan = plan
    return new_df
