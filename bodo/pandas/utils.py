import functools
import importlib
import inspect
import warnings

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
    return _reconstruct_pandas_index(out_df, arrow_schema)


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
                        "Falling back to Pandas (may be slow or run out of memory."
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

    def __str__(self):
        out = f"{self.plan_class}: \n"
        for arg in self.args:
            if isinstance(arg, pd.DataFrame):
                out += f"  {arg.columns.tolist()}\n"
                continue
            out += f"  {arg}\n"
        for k, v in self.kwargs.items():
            out += f"  {k}: {v}\n"
        return out

    __repr__ = __str__

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
            elif isinstance(x, (tuple, list)):
                return type(x)(recursive_check(i) for i in x)
            else:
                return x

        # Convert any LazyPlan in the args or kwargs.
        args = [recursive_check(x) for x in self.args]
        kwargs = {k: recursive_check(v) for k, v in self.kwargs.items()}

        # Create real duckdb class.
        pa_schema = pa.Schema.from_pandas(
            self.empty_data
        )  # do this in filter case? preserve_index=(self.plan_class == "LogicalFilter")
        ret = getattr(plan_optimizer, self.plan_class)(pa_schema, *args, **kwargs)
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

        output_func = cpp_table_to_series if plan.is_series else cpp_table_to_df
        return plan_optimizer.py_execute_plan(
            optimized_plan, output_func, duckdb_plan.out_schema
        )

    if bodo.dataframe_library_run_parallel:
        import bodo.spawn.spawner

        return bodo.spawn.spawner.submit_func_to_workers(_exec_plan, [], plan)

    return _exec_plan(plan)


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


@intrinsic
def cast_table_ptr_to_int64(typingctx, val):
    """Cast C++ table pointer to int64 (to pass to C++ later)"""

    def codegen(context, builder, signature, args):
        return builder.ptrtoint(args[0], lir.IntType(64))

    return numba.core.types.int64(table_type), codegen


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

    n_table_cols = len(df.columns)
    n_index_arrs = get_n_index_arrays(df.index)
    n_all_cols = n_table_cols + n_index_arrs
    in_col_inds = bodo.utils.typing.MetaType(tuple(range(n_all_cols)))

    @numba.jit
    def impl_df_to_cpp_table(df):
        table = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df)
        index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)
        index_arrs = bodo.utils.conversion.index_to_array_list(index, False)
        cpp_table = bodo.libs.array.py_data_to_cpp_table(
            table, index_arrs, in_col_inds, n_table_cols
        )
        return cast_table_ptr_to_int64(cpp_table)

    cpp_table = impl_df_to_cpp_table(df)
    plan_optimizer.set_cpp_table_meta(cpp_table, pa.Schema.from_pandas(df))
    return cpp_table


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


def run_func_on_table(cpp_table, arrow_schema, in_args):
    """Run a user-defined function (UDF) on a DataFrame created from C++ table and
    return the result as a C++ table and column names.
    """
    input = cpp_table_to_df(cpp_table, arrow_schema)
    func_path_str, is_series, is_method, args, kwargs = in_args

    if is_series:
        assert input.shape[1] == 1, "run_func_on_table: single column expected"
        input = input.iloc[:, 0]

    if is_method:
        func = input
        for atr in func_path_str.split("."):
            func = getattr(func, atr)
        out = func(*args, **kwargs)
    else:
        # TODO: test this path
        func = _get_function_from_path(func_path_str)
        out = func(input, *args, **kwargs)

    out_df = pd.DataFrame({"OUT": out})

    # TODO [BSE-4788]: replace with convert_to_arrow_dtypes util
    out_df = out_df.convert_dtypes(dtype_backend="pyarrow")
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
    assert (
        isinstance(proj, LazyPlan)
        and proj.plan_class == "LogicalProjection"
        and len(proj.args[1]) == 1
    ), "get_proj_expr_single: LogicalProjection with a single expr expected"
    return proj.args[1][0]


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
    pa_schema = pa.Schema.from_pandas(src_plan.empty_data)
    exprs = []
    for k in key_indices:
        # Using Arrow schema instead of zero_size_self.iloc to handle Index
        # columns correctly.
        empty_data = arrow_to_empty_df(pa.schema([pa_schema[k]]))
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


def _empty_pd_array(pa_type):
    """Create an empty pandas array with the given Arrow type."""

    # Workaround Arrows conversion gaps for dictionary types
    if isinstance(pa_type, pa.DictionaryType):
        assert pa_type.index_type == pa.int32() and (
            pa_type.value_type == pa.string() or pa_type.value_type == pa.large_string()
        ), "Invalid dictionary type"
        return pd.array(
            ["dummy"], pd.ArrowDtype(pa.dictionary(pa.int32(), pa.string()))
        )[:0]

    pa_arr = pa.array([], type=pa_type, from_pandas=True)
    return pd.array(pa_arr, dtype=pd.ArrowDtype(pa_type))


def arrow_to_empty_df(arrow_schema):
    """Create an empty dataframe with the same schema as the Arrow schema"""
    empty_df = pd.DataFrame(
        {field.name: _empty_pd_array(field.type) for field in arrow_schema}
    )
    return _reconstruct_pandas_index(empty_df, arrow_schema)


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
