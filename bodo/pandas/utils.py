import functools
import importlib
import inspect
import sys
import traceback
import warnings

import pandas as pd
import pyarrow as pa

import bodo
from bodo.pandas.array_manager import LazyArrayManager, LazySingleArrayManager
from bodo.pandas.managers import LazyBlockManager, LazySingleBlockManager
from bodo.utils.typing import check_unsupported_args_fallback

BODO_NONE_DUMMY = "_bodo_none_dummy_"


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


def cpp_table_to_df(cpp_table, arrow_schema=None):
    """Convert a C++ table (table_info) to a pandas DataFrame."""
    from bodo.ext import plan_optimizer

    arrow_table = plan_optimizer.cpp_table_to_arrow(cpp_table)
    df = arrow_table_to_pandas(arrow_table, arrow_schema)
    return df


def cpp_table_to_series(cpp_table, arrow_schema):
    """Convert a C++ table (table_info) to a pandas Series."""
    as_df = cpp_table_to_df(cpp_table, arrow_schema)
    return as_df.iloc[:, 0]


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


@functools.lru_cache
def get_series_string_overloads():
    """Return a list of the functions supported on BodoStringMethods objects
    to some degree by bodo.jit.
    """
    from bodo.hiframes.series_str_impl import SeriesStrMethodType
    from bodo.numba_compat import get_method_overloads

    return get_method_overloads(SeriesStrMethodType)


@functools.lru_cache
def get_groupby_overloads():
    """Return a list of the functions supported on DataFrameGroupby/DataFrameSeries objects
    to some degree by bodo.jit.
    """
    from bodo.hiframes.pd_groupby_ext import DataFrameGroupByType
    from bodo.numba_compat import get_method_overloads

    return get_method_overloads(DataFrameGroupByType)


def get_overloads(cls_name):
    """Use the class name of the __class__ attr of self parameter
    to determine which of the above two functions to call to
    get supported overloads for the current data type.
    """
    if cls_name == "BodoDataFrame":
        return get_dataframe_overloads()
    elif cls_name == "BodoSeries":
        return get_series_overloads()
    elif cls_name == "BodoStringMethods":
        return get_series_string_overloads()
    elif cls_name in ("DataFrameGroupBy", "SeriesGroupBy"):
        return get_groupby_overloads()
    else:
        assert False


class BodoLibNotImplementedException(Exception):
    """Exception raised in the Bodo library when a functionality is not implemented yet
    and we need to fall back to Pandas (captured by the fallback decorator).
    """


class BodoLibFallbackWarning(Warning):
    """Warning raised in the Bodo library in the fallback decorator when some
    functionality is not implemented yet and we need to fall back to Pandas.
    """


def check_args_fallback(
    unsupported=None,
    supported=None,
    package_name="pandas",
    fn_str=None,
    module_name="",
    disable=False,
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
        disable - if True, falls back immediately to the Pandas implementation (used
                in frontend methods that are not fully implemented yet)
    """
    assert (unsupported is None) ^ (supported is None), (
        "Exactly one of unsupported and supported must be specified."
    )

    def decorator(func):
        # See if function is top-level or not by looking for a . in
        # the full name.
        toplevel = "." not in func.__qualname__
        if not bodo.dataframe_library_enabled or disable:
            # Dataframe library not enabled so just call the Pandas super class version.
            if toplevel:
                py_pkg = importlib.import_module(package_name)

                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    # Call the same method in the base class.
                    return getattr(py_pkg, func.__name__)(*args, **kwargs)
            else:

                @functools.wraps(func)
                def wrapper(self, *args, **kwargs):
                    # Call the same method in the base class.
                    return getattr(self.__class__.__bases__[0], func.__name__)(
                        self, *args, **kwargs
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
                    inverted = False
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
                    except_msg = ""
                    if not error:
                        try:
                            return func(*args, **kwargs)
                        except BodoLibNotImplementedException as e:
                            # Fall back to Pandas below
                            except_msg = str(e)
                    # Can we do a top-level override check?

                    # Fallback to Python. Call the same method in the base class.
                    msg = (
                        f"{func.__name__} is not "
                        "implemented in Bodo dataframe library for the specified arguments yet. "
                        "Falling back to Pandas (may be slow or run out of memory)."
                    )
                    if except_msg:
                        msg += f"\nException: {except_msg}"
                    warnings.warn(BodoLibFallbackWarning(msg))
                    return getattr(py_pkg, func.__name__)(*args, **kwargs)
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
                    except_msg = ""
                    if not error:
                        try:
                            return func(self, *args, **kwargs)
                        except BodoLibNotImplementedException as e:
                            # Fall back to Pandas below
                            except_msg = str(e)

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
                    if self.__class__.__name__ in ("DataFrameGroupBy", "SeriesGroupBy"):
                        obj_base_class = self._obj.__class__.__bases__[0]
                        self = getattr(obj_base_class, "groupby")(
                            self._obj,
                            self._keys,
                            as_index=self._as_index,
                            dropna=self._dropna,
                        )[self._selection]
                        base_class = self.__class__
                    elif self.__class__ == bodo.pandas.series.BodoStringMethods:
                        base_class = self._series.__class__.__bases__[0].str
                    else:
                        base_class = self.__class__.__bases__[0]
                    msg = (
                        f"{base_class.__name__}.{func.__name__} is not "
                        "implemented in Bodo dataframe library for the specified arguments yet. "
                        "Falling back to Pandas (may be slow or run out of memory)."
                    )
                    if except_msg:
                        msg += f"\nException: {except_msg}"
                    warnings.warn(BodoLibFallbackWarning(msg))
                    return getattr(base_class, func.__name__)(self, *args, **kwargs)

        return wrapper

    return decorator


class LazyPlan:
    """Easiest mode to use DuckDB is to generate isolated queries and try to minimize
    node re-use issues due to the frequent use of unique_ptr.  This class should be
    used when constructing all plans and holds them lazily.  On demand, generate_duckdb
    can be used to convert to an isolated set of DuckDB objects for execution.
    """

    def __init__(self, plan_class, empty_data, *args, **kwargs):
        self.plan_class = plan_class
        self.args = args
        self.kwargs = kwargs
        assert isinstance(empty_data, (pd.DataFrame, pd.Series)), (
            "LazyPlan: empty_data must be a DataFrame or Series"
        )
        self.is_series = isinstance(empty_data, pd.Series)
        self.empty_data = empty_data
        if self.is_series:
            # None name doesn't round-trip to dataframe correctly so we use a dummy name
            # that is replaced with None in wrap_plan
            name = BODO_NONE_DUMMY if empty_data.name is None else empty_data.name
            self.empty_data = empty_data.to_frame(name=name)

        if (pa_schema := kwargs.get("__pa_schema", None)) is None:
            # Use the schema of the empty data to create the schema for the plan
            # Passing in a separate schema is useful for some operators that need
            # schema metadat like iceberg.
            pa_schema = pa.Schema.from_pandas(self.empty_data)
        else:
            # We don't want this passing to the operator constructors
            del kwargs["__pa_schema"]
        self.pa_schema = pa_schema

    def __str__(self):
        args = self.args

        # Avoid duplicated plan strings by omitting data_source.
        if self.plan_class == "ColRefExpression":
            col_index = args[1]
            return f"ColRefExpression({col_index})"
        elif self.plan_class == "PythonScalarFuncExpression":
            func_name, col_indices = args[1][0], args[2]
            return f"PythonScalarFuncExpression({func_name}, {col_indices})"

        out = f"{self.plan_class}: \n"
        args_str = ""
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                args_str += f"{arg.columns.tolist()}\n"
            elif arg is not None:
                args_str += f"{arg}\n"

        for k, v in self.kwargs.items():
            args_str += f"{k}: {v}\n"

        out += "\n".join(
            f"  {arg_line}"
            for arg_line in args_str.split("\n")
            if not arg_line.isspace()
        )

        return out

    __repr__ = __str__

    def generate_duckdb(self, cache=None):
        from bodo.ext import plan_optimizer

        # Sometimes the same LazyPlan object is encountered twice during the same
        # query so  we use the cache dict to only convert it once.
        if cache is None:
            cache = {}
        # If previously converted then use the last result.
        # Don't cache expression nodes.
        # TODO - Try to eliminate caching altogether since it seems to cause
        # more problems than lack of caching.
        if "Expression" not in self.plan_class and id(self) in cache:
            return cache[id(self)]

        def recursive_check(x):
            """Recursively convert LazyPlans but return other types unmodified."""
            if isinstance(x, LazyPlan):
                return x.generate_duckdb(cache=cache)
            elif isinstance(x, (tuple, list)):
                return type(x)(recursive_check(i) for i in x)
            else:
                return x

        # Convert any LazyPlan in the args or kwargs.
        # We do this in reverse order because we expect the first arg to be
        # the source of the plan and for the node being created to take
        # ownership of that source.  If other args or kwargs reference that
        # plan then if we process them after we have taken ownership then
        # we will get nullptr exceptions.  So, process the args that don't
        # claim ownership first (in the reverse direction) and finally
        # process the first arg which we expect will take ownership.
        kwargs = {k: recursive_check(v) for k, v in self.kwargs.items()}
        args = [recursive_check(x) for x in reversed(self.args)]
        args.reverse()

        # Create real duckdb class.
        ret = getattr(plan_optimizer, self.plan_class)(self.pa_schema, *args, **kwargs)
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

        if (
            bodo.dataframe_library_dump_plans
            and bodo.libs.distributed_api.get_rank() == 0
        ):
            # Sometimes when an execution is triggered it isn't expected that
            # an execution should happen at that point.  This traceback is
            # useful to identify what is triggering the execution as it may be
            # a bug or the usage of some Pandas API that calls a function that
            # triggers execution.  This traceback can help fix the bug or
            # select a different Pandas API or an internal Pandas function that
            # bypasses the issue.
            traceback.print_stack(file=sys.stdout)
            print("")  # Print on new line during tests.
            print("Unoptimized plan")
            print(duckdb_plan.toString())

        # Print the plan before optimization
        if bodo.tracing_level >= 2 and bodo.libs.distributed_api.get_rank() == 0:
            pre_optimize_graphviz = duckdb_plan.toGraphviz()
            with open("pre_optimize" + str(id(plan)) + ".dot", "w") as f:
                print(pre_optimize_graphviz, file=f)

        optimized_plan = plan_optimizer.py_optimize_plan(duckdb_plan)

        if (
            bodo.dataframe_library_dump_plans
            and bodo.libs.distributed_api.get_rank() == 0
        ):
            print("Optimized plan")
            print(optimized_plan.toString())

        # Print the plan after optimization
        if bodo.tracing_level >= 2 and bodo.libs.distributed_api.get_rank() == 0:
            post_optimize_graphviz = optimized_plan.toGraphviz()
            with open("post_optimize" + str(id(plan)) + ".dot", "w") as f:
                print(post_optimize_graphviz, file=f)

        output_func = cpp_table_to_series if plan.is_series else cpp_table_to_df
        return plan_optimizer.py_execute_plan(
            optimized_plan, output_func, duckdb_plan.out_schema
        )

    if bodo.dataframe_library_run_parallel:
        import bodo.spawn.spawner

        return bodo.spawn.spawner.submit_func_to_workers(_exec_plan, [], plan)

    return _exec_plan(plan)


def get_plan_cardinality(plan: LazyPlan):
    """See if we can statically know the cardinality of the result of the plan.

    Args:
        plan (LazyPlan): query plan to get cardinality of.

    Returns:
        int (if cardinality is known) or None (if not known)
    """

    duckdb_plan = plan.generate_duckdb()
    return duckdb_plan.getCardinality()


def getPlanStatistics(plan: LazyPlan):
    """Get statistics for a plan pre and post optimization.

    Args:
        plan (LazyPlan): query plan to get statistics for

    Returns:
        Number of nodes in the tree before and after optimization.
    """
    from bodo.ext import plan_optimizer

    duckdb_plan = plan.generate_duckdb()
    preOptNum = plan_optimizer.count_nodes(duckdb_plan)
    optimized_plan = plan_optimizer.py_optimize_plan(duckdb_plan)
    postOptNum = plan_optimizer.count_nodes(optimized_plan)
    return preOptNum, postOptNum


def get_n_index_arrays(index):
    """Get the number of arrays that can hold the Index data in a table."""
    if isinstance(index, pd.RangeIndex):
        return 0
    elif isinstance(index, pd.MultiIndex):
        return index.nlevels
    elif isinstance(index, pd.Index):
        return 1
    else:
        raise TypeError(f"Invalid index type: {type(index)}")


def df_to_cpp_table(df):
    """Convert a pandas DataFrame to a C++ table pointer with column names and
    metadata set properly.
    """
    from bodo.ext import plan_optimizer

    # TODO: test nthreads, safe
    arrow_table = pa.Table.from_pandas(df)
    return plan_optimizer.arrow_to_cpp_table(arrow_table)


def _empty_pd_array(pa_type):
    """Create an empty pandas array with the given Arrow type."""

    # Workaround Arrows conversion gaps for dictionary types
    if isinstance(pa_type, pa.DictionaryType):
        assert pa_type.index_type == pa.int32() and (
            pa_type.value_type == pa.string() or pa_type.value_type == pa.large_string()
        ), (
            "Invalid dictionary type "
            + str(pa_type.index_type)
            + " "
            + str(pa_type.value_type)
        )
        return pd.array(
            ["dummy"], pd.ArrowDtype(pa.dictionary(pa.int32(), pa.string()))
        )[:0]

    pa_arr = pa.array([], type=pa_type, from_pandas=True)
    return pd.array(pa_arr, dtype=pd.ArrowDtype(pa_type))


def _get_function_from_path(path_str: str):
    """Get a function object from its fully qualified path string.

    Args:
        path_str (str): The function path in format 'module.submodule.function'

    Returns:
        callable: The function object

    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the function doesn't exist in the module
    """
    parts = path_str.split(".")
    module_path = ".".join(parts[:-1])
    func_name = parts[-1]

    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def run_func_on_table(cpp_table, result_type, in_args):
    """Run a user-defined function (UDF) on a DataFrame created from C++ table and
    return the result as a C++ table and column names.
    """
    input = cpp_table_to_df(cpp_table)
    func, is_series, is_attr, args, kwargs = in_args

    if is_series:
        assert input.shape[1] == 1, "run_func_on_table: single column expected"
        input = input.iloc[:, 0]

    if isinstance(func, str) and is_attr:
        func_path_str = func
        func = input
        for atr in func_path_str.split("."):
            func = getattr(func, atr)
        if not callable(func):
            # func is assumed to be an accessor
            out = func
        else:
            out = func(*args, **kwargs)
    elif isinstance(func, str):
        func = _get_function_from_path(func)
        out = func(input, *args, **kwargs)
    else:
        out = func(input, *args, **kwargs)

    # astype can fail in some cases when input is empty
    if len(out):
        # TODO: verify this is correct for all possible result_type's
        out_df = pd.DataFrame({"OUT": out.astype(pd.ArrowDtype(result_type))})
    else:
        out_df = pd.DataFrame({"OUT": _empty_pd_array(result_type)})

    return df_to_cpp_table(out_df)


def _del_func(x):
    # Intentionally do nothing
    pass


def _get_index_data(index):
    """Get the index data from a pandas Index object to be passed to BodoDataFrame or
    BodoSeries.
    Roughly similar to spawn worker handling of Index:
    https://github.com/bodo-ai/Bodo/blob/452ba4c5f18fcc531822827f1aed0e212b09c595/bodo/spawn/worker.py#L124
    """
    from pandas.core.arrays.arrow import ArrowExtensionArray

    if isinstance(index, pd.RangeIndex):
        data = None
    elif isinstance(index, pd.MultiIndex):
        data = index.to_frame(index=False, allow_duplicates=True)
    elif isinstance(index, pd.Index):
        data = ArrowExtensionArray(pa.array(index._data))
    else:
        raise TypeError(f"Invalid index type: {type(index)}")

    return data


def wrap_plan(plan, res_id=None, nrows=None):
    """Create a BodoDataFrame or BodoSeries with the given
    schema and given plan node.
    """

    from bodo.pandas.frame import BodoDataFrame
    from bodo.pandas.lazy_metadata import LazyMetadata
    from bodo.pandas.series import BodoSeries
    from bodo.pandas.utils import (
        LazyPlan,
        get_lazy_manager_class,
        get_lazy_single_manager_class,
    )

    assert isinstance(plan, LazyPlan), "wrap_plan: LazyPlan expected"

    if nrows is None:
        # Fake non-zero rows. nrows should be overwritten upon plan execution.
        nrows = 1

    index_data = _get_index_data(plan.empty_data.index)

    if not plan.is_series:
        metadata = LazyMetadata(
            res_id,
            plan.empty_data,
            nrows=nrows,
            index_data=index_data,
        )
        mgr = get_lazy_manager_class()
        new_df = BodoDataFrame.from_lazy_metadata(
            metadata, collect_func=mgr._collect, del_func=_del_func, plan=plan
        )
    else:
        empty_data = plan.empty_data.squeeze()
        # Replace the dummy name with None set in LazyPlan constructor
        if empty_data.name == BODO_NONE_DUMMY:
            empty_data.name = None
        metadata = LazyMetadata(
            res_id,
            empty_data,
            nrows=nrows,
            index_data=index_data,
        )
        mgr = get_lazy_single_manager_class()
        new_df = BodoSeries.from_lazy_metadata(
            metadata, collect_func=mgr._collect, del_func=_del_func, plan=plan
        )

    return new_df


def get_proj_expr_single(proj: LazyPlan):
    """Get the single expression from a LogicalProjection node."""
    if is_single_projection(proj):
        return proj.args[1][0]
    else:
        if not proj.is_series:
            raise Exception("Got a non-Series in get_proj_expr_single")
        return make_col_ref_exprs([0], proj)[0]


def get_single_proj_source_if_present(proj: LazyPlan):
    """Get the single expression from a LogicalProjection node."""
    if is_single_projection(proj):
        return proj.args[0]
    else:
        if not proj.is_series:
            raise Exception("Got a non-Series in get_single_proj_source_if_present")
        return proj


def is_single_projection(proj: LazyPlan):
    """Return True if plan is a projection with a single expression"""
    return (
        isinstance(proj, LazyPlan)
        and proj.plan_class == "LogicalProjection"
        and len(proj.args[1]) == (get_n_index_arrays(proj.empty_data.index) + 1)
    )


def is_single_colref_projection(proj: LazyPlan):
    """Return True if plan is a projection with a single expression that is a column reference"""
    return (
        is_single_projection(proj) and proj.args[1][0].plan_class == "ColRefExpression"
    )


def make_col_ref_exprs(key_indices, src_plan):
    """Create column reference expressions for the given key indices for the input
    source plan.
    """

    exprs = []
    for k in key_indices:
        # Using Arrow schema instead of zero_size_self.iloc to handle Index
        # columns correctly.
        empty_data = arrow_to_empty_df(pa.schema([src_plan.pa_schema[k]]))
        p = LazyPlan("ColRefExpression", empty_data, src_plan, k)
        exprs.append(p)

    return exprs


def _is_generated_index_name(name):
    """Check if the Index name is a generated name similar to PyArrow:
    https://github.com/apache/arrow/blob/5e9fce493f21098d616f08034bc233fcc529b3ad/python/pyarrow/pandas_compat.py#L1071
    """
    import re

    pattern = r"^__index_level_\d+__$"
    return re.match(pattern, name) is not None


def _reconstruct_pandas_index(df, arrow_schema):
    """Reconstruct the pandas Index from the metadata in Arrow schema (some columns may
    be moved to Index/MultiIndex).
    Similar to PyArrow, but simpler since we don't support all backward compatibility:
    https://github.com/apache/arrow/blob/5e9fce493f21098d616f08034bc233fcc529b3ad/python/pyarrow/pandas_compat.py#L974
    """

    if arrow_schema.pandas_metadata is None:
        return df

    index_arrays = []
    index_names = []
    for descr in arrow_schema.pandas_metadata.get("index_columns", []):
        if isinstance(descr, str):
            index_name = None if _is_generated_index_name(descr) else descr
            index_level = df[descr]
            df = df.drop(columns=[descr])
        elif descr["kind"] == "range":
            index_name = descr["name"]
            start = descr["start"]
            step = descr["step"]
            # Set stop value to proper size since we create PyArrow schema from empty
            # DataFrames
            stop = start + step * len(df)
            index_level = pd.RangeIndex(start, stop, step, name=index_name)
        else:
            raise ValueError(f"Unrecognized index kind: {descr['kind']}")
        index_arrays.append(index_level)
        index_names.append(index_name)

    # Reconstruct the row index
    if len(index_arrays) > 1:
        index = pd.MultiIndex.from_arrays(index_arrays, names=index_names)
    elif len(index_arrays) == 1:
        index = index_arrays[0]
        if not isinstance(index, pd.Index):
            # Box anything that wasn't boxed above
            index = pd.Index(index)
            # Setting name outside of the constructor since it prioritizes Series name
            # from input Series.
            index.name = index_names[0]
    else:
        index = pd.RangeIndex(len(df))

    df.index = index
    return df


def arrow_to_empty_df(arrow_schema):
    """Create an empty dataframe with the same schema as the Arrow schema"""
    empty_df = pd.DataFrame(
        {field.name: _empty_pd_array(field.type) for field in arrow_schema}
    )
    return _reconstruct_pandas_index(empty_df, arrow_schema)


def _fix_struct_arr_names(arr, pa_type):
    """Fix the names of the fields in a struct array to match the Arrow type.
    This is necessary since our C++ code may not preserve the field names in
    struct arrays.
    """

    if not pa.types.is_struct(arr.type):
        return arr

    if arr.type == pa_type:
        return arr

    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()

    new_arrs = [
        _fix_struct_arr_names(arr.field(i), pa_type.field(i).type)
        for i in range(arr.type.num_fields)
    ]
    names = [pa_type.field(i).name for i in range(pa_type.num_fields)]
    new_arr = pa.StructArray.from_arrays(new_arrs, names)
    # Arrow's from_arrays ignores nulls (bug as of Arrow 19) so we add them back
    # manually
    return pa.Array.from_buffers(
        new_arr.type, len(new_arr), arr.buffers()[:1], children=new_arrs
    )


def _arrow_to_pd_array(arrow_array, pa_type):
    """Convert a PyArrow array to a pandas array with the specified Arrow type."""

    # Our type inference may fail for some object columns so use the proper Arrow type
    if pa_type == pa.null():
        pa_type = arrow_array.type

    # Our C++ code may not preserve the field names in struct arrays
    # so we fix them here to match the Arrow schema.
    if pa.types.is_struct(arrow_array.type):
        arrow_array = _fix_struct_arr_names(arrow_array, pa_type)

    # Cast to expected type to match Pandas (as determined by the frontend)
    if pa_type != arrow_array.type:
        arrow_array = arrow_array.cast(pa_type)

    return pd.array(arrow_array, dtype=pd.ArrowDtype(pa_type))


def arrow_table_to_pandas(arrow_table, arrow_schema=None):
    """Convert a PyArrow Table to a pandas DataFrame. Not using Table.to_pandas()
    since it doesn't use ArrowDtype and has issues (e.g. repeated column names fails).

    Args:
        arrow_table (pa.Table): The input Arrow table.
        arrow_schema (pa.Schema, optional): The schema to use for the DataFrame.
            If None, uses the schema from the Arrow table.

    Returns:
        pd.DataFrame: The converted pandas DataFrame.
    """
    if arrow_schema is None:
        arrow_schema = arrow_table.schema

    df = pd.DataFrame(
        {
            i: _arrow_to_pd_array(arrow_table.columns[i], field.type)
            for i, field in enumerate(arrow_schema)
        }
    )
    # Set column names separately to handle duplicate names ("field.name:" in a
    # dictionary would replace duplicated values)
    df.columns = [f.name for f in arrow_schema]

    df_with_index = _reconstruct_pandas_index(df, arrow_schema)

    # Handle multi-level column names e.g. ["('A', 'sum')", "('A', 'mean')"]
    if (
        arrow_schema.pandas_metadata is not None
        and len(arrow_schema.pandas_metadata.get("column_indexes", [])) > 1
    ):
        columns_zipped = zip(*[eval(col) for col in df_with_index.columns])
        df_with_index.columns = pd.MultiIndex.from_arrays(columns_zipped)

    return df_with_index


def _get_empty_series_arrow(ser: pd.Series) -> pd.Series:
    """Create an empty Series like ser possibly converting some dtype to use
    pyarrow"""
    empty_df = arrow_to_empty_df(pa.Schema.from_pandas(ser.to_frame()))
    empty_series = empty_df.squeeze()
    empty_series.name = ser.name
    return empty_series


def get_scalar_udf_result_type(obj, method_name, func, *args, **kwargs) -> pd.Series:
    """Infer the output type of a scalar UDF by running it on a
    sample of the data.

    Args:
        obj (BodoDataFrame | BodoSeries): The object the UDF is being applied over.
        method_name ({"apply", "map", "map_parititons"}): The name of the method
            applying the UDF.
        func (Any): The UDF argument to pass to apply/map.
        kwargs (dict): Optional keyword arguments to pass to apply/map.

    Raises:
        BodoLibNotImplementedException: If the dtype cannot be infered.

    Returns:
        Empty Series with the dtype matching the output of the UDF
        (or equivalent pyarrow dtype)
    """
    assert method_name in {"map", "apply", "map_partitions"}, (
        "expected method to be one of {'apply', 'map', 'map_partitions'}"
    )

    base_class = obj.__class__.__bases__[0]

    # map_partitions is not a pandas.DataFrame method.
    apply_method = None
    if method_name != "map_partitions":
        apply_method = getattr(base_class, method_name)

    # TODO: Tune sample sizes
    sample_sizes = (1, 4, 9, 25, 100)

    except_msg = ""
    for sample_size in sample_sizes:
        df_sample = obj.head(sample_size).execute_plan()
        pd_sample = base_class(df_sample)
        out_sample = (
            func(pd_sample, *args, **kwargs)
            if apply_method is None
            else apply_method(pd_sample, func, *args, **kwargs)
        )

        if not isinstance(out_sample, pd.Series):
            raise BodoLibNotImplementedException(
                f"expected output to be Series, got: {type(out_sample)}."
            )

        # For Series.map with na_action='ignore' and NA values in the first rows,
        # the type infered will be the type of the NA, not necessarily the actual
        # return type.
        if not pd.isna(out_sample).all():
            try:
                empty_series = _get_empty_series_arrow(out_sample)
            except (pa.lib.ArrowTypeError, pa.lib.ArrowInvalid) as e:
                # Could not get a pyarrow type for the series, Fallback to pandas.
                except_msg = f", got: {str(e)}."
                break

            return empty_series

        # all the data was collected and couldn't infer types,
        # fall back to pandas.
        if len(out_sample) < sample_size:
            break

        # TODO: Warning that repeated sampling may hurt performance.

    raise BodoLibNotImplementedException(
        f"could not infer the output type of user defined function{except_msg}."
    )


class LazyPlanDistributedArg:
    """
    Class to hold the arguments for a LazyPlan that are distributed on the workers.
    """

    def __init__(self, mgr, res_id: str):
        self.mgr = mgr
        self.res_id = res_id

    def __reduce__(self):
        """
        This method is used to serialize the object for distribution.
        We can't send the manager to the workers without triggering collection
        so we just send the result ID instead.
        """
        return (str, (self.res_id,))


def count_plan(self):
    # See if we can get the cardinality statically.
    static_cardinality = get_plan_cardinality(self._plan)
    if static_cardinality is not None:
        return static_cardinality

    # Can't be known statically so create count plan on top of
    # existing plan.
    count_star_schema = pd.Series(dtype="uint64", name="count_star")
    aggregate_plan = LazyPlan(
        "LogicalAggregate",
        count_star_schema,
        self._plan,
        [],
        [
            LazyPlan(
                "AggregateExpression",
                count_star_schema,
                self._plan,
                "count_star",
                # Adding column 0 as input to avoid deleting all input by the optimizer
                # TODO: avoid materializing the input column
                [0],
                False,  # dropna
            )
        ],
    )
    projection_plan = LazyPlan(
        "LogicalProjection",
        count_star_schema,
        aggregate_plan,
        make_col_ref_exprs([0], aggregate_plan),
    )

    data = execute_plan(projection_plan)
    return data[0]


def ensure_datetime64ns(df):
    """Convert datetime columns in a DataFrame to 'datetime64[ns]' dtype.
    Avoids datetime64[us] that is commonly used in Pandas but not supported in Bodo.
    """
    import numpy as np

    for c in df.columns:
        dtype = df[c].dtype
        if (
            isinstance(dtype, np.dtype)
            and dtype.kind == "M"
            and dtype.name != "datetime64[ns]"
        ):
            df[c] = df[c].astype("datetime64[ns]")

    if (
        isinstance(df.index, pd.DatetimeIndex)
        and isinstance(df.index.dtype, np.dtype)
        and df.index.dtype.kind == "M"
        and df.index.dtype.name != "datetime64[ns]"
    ):
        df.index = df.index.astype("datetime64[ns]")

    return df


def _get_df_python_func_plan(df_plan, empty_data, func, args, kwargs, is_method=True):
    """Create plan for calling some function or method on a DataFrame. Creates a
    PythonScalarFuncExpression with provided arguments and a LogicalProjection.
    """
    df_len = len(df_plan.empty_data.columns)
    udf_arg = LazyPlan(
        "PythonScalarFuncExpression",
        empty_data,
        df_plan,
        (
            func,
            False,  # is_series
            is_method,
            args,
            kwargs,
        ),
        tuple(range(df_len + get_n_index_arrays(df_plan.empty_data.index))),
    )

    # Select Index columns explicitly for output
    index_col_refs = tuple(
        make_col_ref_exprs(
            range(df_len, df_len + get_n_index_arrays(df_plan.empty_data.index)),
            df_plan,
        )
    )
    plan = LazyPlan(
        "LogicalProjection",
        empty_data,
        df_plan,
        (udf_arg,) + index_col_refs,
    )
    return wrap_plan(plan=plan)


def is_col_ref(expr):
    return expr.plan_class == "ColRefExpression"


def is_scalar_func(expr):
    return expr.plan_class == "PythonScalarFuncExpression"


def is_arith_expr(expr):
    return expr.plan_class == "ArithOpExpression"
