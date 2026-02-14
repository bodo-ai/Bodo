from __future__ import annotations

import atexit
import functools
import importlib
import inspect
import time
import types as pytypes
import typing as pt
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa

import bodo
from bodo.pandas.array_manager import LazyArrayManager, LazySingleArrayManager
from bodo.pandas.managers import LazyBlockManager, LazySingleBlockManager

if pt.TYPE_CHECKING:
    from bodo.pandas.plan import ColRefExpression, LazyPlan
    from bodo.pandas.scalar import BodoScalar

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


def normalize_slice_indices_for_lazy_md(
    slobj: slice, nrows: int
) -> tuple[int, int | None, int]:
    """Normalize negative/None start/stop/step for slicing lazy metadata.

    Args:
        slobj (slice): The slice object
        nrows (int): Total number of rows in the DataFrame/Series

    Returns:
        tuple[int, int | None, int]: A tuple of normalized (start, stop, step)
            without negative start/stop indices. stop is None implies the slice
            goes from start to the beginning in reverse order.
    """
    start, stop, step = slobj.indices(nrows)
    stop = None if stop < 0 and step < 0 else stop
    return start, stop, step


def schema_has_index_arrays(arrow_schema: pa.Schema) -> bool:
    """Return True if the Arrow schema has index arrays, False otherwise
    (RangeIndex case).

    Args:
        arrow_schema (pa.Schema): The Arrow schema to check.

    Returns:
        bool: True if the schema has index arrays, False otherwise.
    """

    if arrow_schema.pandas_metadata is None:
        return False

    for descr in arrow_schema.pandas_metadata.get("index_columns", []):
        if isinstance(descr, str):
            if descr not in arrow_schema.names:
                # Index not found in table: matching Pyarrow's behavior, which treats
                # missing index as RangeIndex.
                continue
            return True
        elif descr["kind"] == "range":
            continue
        else:
            raise ValueError(f"Unrecognized index kind: {descr['kind']}")

    return False


def convert_to_pandas_types(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Returns a DataFrame with the same columns with arrow types cast to
    Pandas to avoid limitations in Pandas arrow types casting.
    """

    def dtype_to_pandas(dtype):
        if isinstance(dtype, pd.ArrowDtype):
            return pa.array([], dtype.pyarrow_dtype).to_pandas().dtype
        return dtype

    if isinstance(obj, pd.Series):
        new_type = dtype_to_pandas(obj.dtype)
        return obj.astype(new_type)

    new_obj = pd.DataFrame()
    for c in obj.columns:
        if not isinstance(obj[c], pd.Series):
            raise BodoLibNotImplementedException(
                "Bodo DataFrame with duplicate columns detected."
            )
        new_type = dtype_to_pandas(obj[c].dtype)
        new_obj[c] = obj[c].astype(new_type)

    return new_obj


def cpp_table_to_df(
    cpp_table, arrow_schema=None, use_arrow_dtypes=True, delete_input=True
):
    """Convert a C++ table (table_info) to a pandas DataFrame."""
    from bodo.ext import plan_optimizer

    arrow_table = plan_optimizer.cpp_table_to_arrow(cpp_table, delete_input)
    df = arrow_table_to_pandas(arrow_table, arrow_schema, use_arrow_dtypes)
    return df


def cpp_table_to_series(
    cpp_table,
    arrow_schema=None,
    use_arrow_dtypes=True,
    ignore_index=False,
    delete_input=True,
):
    """Convert a C++ table (table_info) to a pandas Series."""
    from bodo.ext import plan_optimizer

    # We need to preserve Index for query output case (which provides arrow_schema also,
    # see execute_plan)
    if not ignore_index and schema_has_index_arrays(arrow_schema):
        as_df = cpp_table_to_df(cpp_table, arrow_schema, use_arrow_dtypes, delete_input)
        return as_df.iloc[:, 0]

    arrow_arr, name = plan_optimizer.cpp_table_to_arrow_array(cpp_table, delete_input)
    arrow_type = arrow_arr.type if arrow_schema is None else arrow_schema[0].type

    return _arrow_array_to_pd(arrow_arr, arrow_type, use_arrow_dtypes, name=name)


@functools.lru_cache
def get_dataframe_overloads():
    """Return a list of the functions supported on BodoDataFrame objects
    to some degree by bodo.jit.
    """
    # Import compiler
    import bodo.decorators  # isort:skip # noqa
    from bodo.hiframes.pd_dataframe_ext import DataFrameType
    from bodo.numba_compat import get_method_overloads

    return get_method_overloads(DataFrameType)


@functools.lru_cache
def get_series_overloads():
    """Return a list of the functions supported on BodoSeries objects
    to some degree by bodo.jit.
    """
    # Import compiler
    import bodo.decorators  # isort:skip # noqa
    from bodo.hiframes.pd_series_ext import SeriesType
    from bodo.numba_compat import get_method_overloads

    return get_method_overloads(SeriesType)


@functools.lru_cache
def get_series_string_overloads():
    """Return a list of the functions supported on BodoStringMethods objects
    to some degree by bodo.jit.
    """
    # Import compiler
    import bodo.decorators  # isort:skip # noqa
    from bodo.hiframes.series_str_impl import SeriesStrMethodType
    from bodo.numba_compat import get_method_overloads

    return get_method_overloads(SeriesStrMethodType)


@functools.lru_cache
def get_series_datetime_overloads():
    """Return a list of the functions supported on SeriesDatetimePropertiesType objects
    to some degree by bodo.jit.
    """
    # Import compiler
    import bodo.decorators  # isort:skip # noqa
    from bodo.hiframes.series_dt_impl import SeriesDatetimePropertiesType
    from bodo.numba_compat import get_method_overloads

    return get_method_overloads(SeriesDatetimePropertiesType)


@functools.lru_cache
def get_groupby_overloads():
    """Return a list of the functions supported on DataFrameGroupby/DataFrameSeries objects
    to some degree by bodo.jit.
    """
    # Import compiler
    import bodo.decorators  # isort:skip # noqa
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
    elif cls_name == "BodoDatetimeProperties":
        return get_series_datetime_overloads()
    elif cls_name in ("DataFrameGroupBy", "SeriesGroupBy"):
        return get_groupby_overloads()
    else:
        assert False


class BodoLibNotImplementedException(Exception):
    """Exception raised in the Bodo library when a functionality is not implemented yet
    and we need to fall back to Pandas (captured by the fallback decorator).
    """


class BodoDictionaryTypeInvalidException(Exception):
    """Exception raised in the Bodo DataFrames when unsupported dictionary type is
    encountered (either values are not strings or index type is not int32).
    """


class BodoLibFallbackWarning(Warning):
    """Warning raised in the Bodo library in the fallback decorator when some
    functionality is not implemented yet and we need to fall back to Pandas.
    """


class BodoCompilationFailedWarning(Warning):
    """Warning raised when executing UDFs (apply, map) when compiling and running user
    provided function on empty data raises a BodoError or did not return a valid type.
    """


top_time = 0
method_time = 0


def report_times():
    if bodo.get_rank() == 0:
        print("profile_time atexit total_top_time", top_time)
        print("profile_time atexit total_method_time", method_time)
        print("profile_time atexit total_init_lazy", bodo.pandas.plan.total_init_lazy)
        print(
            "profile_time atexit total_execute_plan",
            bodo.pandas.plan.total_execute_plan,
        )


if bodo.dataframe_library_profile:
    atexit.register(report_times)

if bodo.gpu_enabled:
    from bodo.ext import plan_optimizer
    from bodo.numba_compat import BodoCacheLocator

    plan_optimizer.c_set_use_cudf(True, BodoCacheLocator.cache_dir)


def _maybe_create_bodo_obj(cls, obj: pd.DataFrame | pd.Series):
    """Wrap obj with a Bodo constructor or return obj unchanged if
    it contains invalid Arrow types."""

    try:
        return cls(obj)
    except BodoLibNotImplementedException as e:
        warnings.warn(
            BodoLibFallbackWarning(
                f"Could not convert object to {cls.__name__} during fallback, "
                + f"execution will continue using Pandas: {e}"
            )
        )

    return obj


def convert_to_bodo(obj):
    """Returns a new version of *obj* that is the equivalent Bodo type or leave unchanged
    if not a DataFrame or Series."""
    from bodo.pandas import BodoDataFrame, BodoSeries

    # Avoid converting to Bodo types if dataframe library is disabled for testing
    if not bodo.dataframe_library_enabled:
        return obj

    if isinstance(obj, pd.DataFrame) and not isinstance(obj, BodoDataFrame):
        return _maybe_create_bodo_obj(BodoDataFrame, obj)
    elif isinstance(obj, pd.Series) and not isinstance(obj, BodoSeries):
        return _maybe_create_bodo_obj(BodoSeries, obj)
    return obj


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
                            start_time = time.perf_counter()
                            ret = func(*args, **kwargs)
                            global top_time
                            time_this_call = time.perf_counter() - start_time
                            if bodo.dataframe_library_profile:
                                print(
                                    "profile_time top_level",
                                    func.__qualname__,
                                    time_this_call,
                                )
                            top_time += time_this_call
                            return ret
                        except BodoLibNotImplementedException as e:
                            # Fall back to Pandas below
                            except_msg = str(e)
                    # Can we do a top-level override check?

                    # Fallback to Python. Call the same method in the base class.
                    msg = (
                        f"{func.__name__} is not "
                        "implemented in Bodo DataFrames for the specified arguments yet. "
                        "Falling back to Pandas (may be slow or run out of memory)."
                    )
                    if except_msg:
                        msg += f"\nException: {except_msg}"
                    fallback_warn(msg)
                    py_res = getattr(py_pkg, func.__name__)(*args, **kwargs)
                    return convert_to_bodo(py_res)
            else:

                @functools.wraps(func)
                def wrapper(self, *args, **kwargs):
                    from bodo.pandas import BODO_PANDAS_FALLBACK

                    error = check_unsupported_args_fallback(
                        func.__qualname__,
                        unsupported_args,
                        unsupported_kwargs,
                        (self, *args),
                        kwargs,
                        package_name=package_name,
                        fn_str=fn_str,
                        module_name=module_name,
                        raise_on_error=(BODO_PANDAS_FALLBACK == 0),
                    )
                    except_msg = ""
                    if not error:
                        try:
                            start_time = time.perf_counter()
                            ret = func(self, *args, **kwargs)
                            global method_time
                            time_this_call = time.perf_counter() - start_time
                            if bodo.dataframe_library_profile:
                                print(
                                    "profile_time method",
                                    func.__qualname__,
                                    time_this_call,
                                )
                            method_time += time_this_call
                            return ret
                        except BodoLibNotImplementedException as e:
                            # Fall back to Pandas below
                            except_msg = str(e)

                    # Fallback to Python. Call the same method in the base class.
                    if self.__class__.__name__ in ("DataFrameGroupBy", "SeriesGroupBy"):
                        obj_base_class = self._obj.__class__.__bases__[0]
                        grouped = getattr(obj_base_class, "groupby")(
                            self._obj,
                            self._keys,
                            as_index=self._as_index,
                            dropna=self._dropna,
                        )
                        self = grouped[self._selection] if self._selection else grouped
                        base_class = self.__class__
                    elif self.__class__ == bodo.pandas.series.BodoStringMethods:
                        base_class = self._series.__class__.__bases__[0].str
                        # Workaround Pandas 3 bugs for concat with nulls for Arrow dtype
                        self = pd.Series(self._series).astype(object).str
                    elif self.__class__ == bodo.pandas.series.BodoDatetimeProperties:
                        base_class = self._series.__class__.__bases__[0].dt
                    else:
                        base_class = self.__class__.__bases__[0]
                    msg = (
                        f"{base_class.__name__}.{func.__name__} is not "
                        "implemented in Bodo DataFrames for the specified arguments yet. "
                        "Falling back to Pandas (may be slow or run out of memory)."
                    )
                    if except_msg:
                        msg += f"\nException: {except_msg}"
                    py_res = fallback_wrapper(
                        self, getattr(base_class, func.__name__), func.__name__, msg
                    )(self, *args, **kwargs)
                    return py_res

        return wrapper

    return decorator


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


def df_to_cpp_table(df) -> tuple[int, pa.Schema]:
    """Convert a pandas DataFrame to a C++ table pointer with column names and
    metadata set properly and returns a pointer to the C++ along with the Arrow
    schema of the input DataFrame.
    """
    from bodo.ext import plan_optimizer
    from bodo.pandas.frame import BodoDataFrame

    # Avoid "maximum recursion depth" error in case df is a BodoDataFrame
    if isinstance(df, BodoDataFrame):
        df = pd.DataFrame(df, copy=False)

    arrow_table = pa.Table.from_pandas(df)

    # Ensure all columns have exactly 1 chunk as expected by our C++ code
    arrow_table = arrow_table.combine_chunks()

    # Handle zero chunk cases
    new_columns = []
    for column in arrow_table.columns:
        if column.num_chunks == 0:
            # Create a single empty chunk for zero-chunk columns
            empty_array = pa.array([], type=column.type)
            new_column = pa.chunked_array([empty_array])
            new_columns.append(new_column)
        else:
            new_columns.append(column)

    # Recreate table with fixed columns if any changes were made
    if any(col.num_chunks == 0 for col in arrow_table.columns):
        arrow_table = pa.Table.from_arrays(new_columns, schema=arrow_table.schema)

    return plan_optimizer.arrow_to_cpp_table(arrow_table), arrow_table.schema


def _empty_pd_array(pa_type, field_name=None):
    """Create an empty pandas array with the given Arrow type."""

    # Workaround Arrows conversion gaps for dictionary types
    if isinstance(pa_type, pa.DictionaryType):
        if not (
            pa_type.index_type == pa.int32()
            and (
                pa_type.value_type == pa.string()
                or pa_type.value_type == pa.large_string()
            )
        ):
            field_part = f" at column {field_name}" if field_name is not None else ""
            raise BodoDictionaryTypeInvalidException(
                f"Encountered invalid dictionary type{field_part}: "
                + str(pa_type.index_type)
                + " "
                + str(pa_type.value_type)
                + " not supported yet."
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
    NOTE: needs to free cpp_table after use.
    """
    from bodo.ext import plan_optimizer

    func, is_series, is_attr, args, kwargs, use_arrow_dtypes = in_args

    if use_arrow_dtypes is None:
        # Arrow dtypes can be very slow for UDFs in Pandas:
        # https://github.com/pandas-dev/pandas/issues/61747
        # TODO[BSE-4948]: Use Arrow dtypes when Bodo engine is specified
        # Note: `add`, `sub`, `radd` and `rsub` do not use Arrow dtypes because
        # Arrow does not support element-wise binary operations
        # across most scalar types. Instead, fallback logic using Pandas semantics
        # is used to ensure consistent behavior.
        use_arrow_dtypes = not (
            is_attr
            and func
            in (
                "apply",
                "add",
                "sub",
                "radd",
                "rsub",
            )
        )

    cpp_to_py_start = time.perf_counter_ns()
    if is_series:
        # NOTE: Assuming Series operations ignore Indexes.
        # delete_input=False since cpp_table is needed for output below
        input = cpp_table_to_series(
            cpp_table,
            use_arrow_dtypes=use_arrow_dtypes,
            ignore_index=True,
            delete_input=False,
        )
    else:
        input = cpp_table_to_df(cpp_table, use_arrow_dtypes=use_arrow_dtypes)
    cpp_to_py = (time.perf_counter_ns() - cpp_to_py_start) // 1000

    udf_time_start = time.perf_counter_ns()
    if isinstance(func, str) and is_attr:
        # We implement our BodoSeries map_with_state with an underlying
        # Pandas map and a function whose closure captures the previously
        # created init_state.
        if func == "map_with_state":
            if len(args) != 3:
                raise Exception(
                    f"Got unexpected number of args {len(args)} for map_with_state"
                )
            func = "map"  # Use pandas map to implement map_with_state
            # Extract args[1] which is the mapping function provided by the user
            # as row_fn in map_with_state.  We have to do these extractions and
            # not use them directly in state_wrapper below because we reuse args
            # and that changes the closure capture you'd otherwise expect in
            # state_wrapper.
            state_wrapper_func = args[1]
            # Extract args[0] which is the previously created init_state.
            state_wrapper_state = args[0]

            def state_wrapper(x):
                # Call the user-provided row_fn function passing the init_state
                # and the row x from the table.
                return state_wrapper_func(state_wrapper_state, x)

            # Map takes two args, the function to run and args[2] which is na_action.
            args = (state_wrapper, args[2])
        elif func == "map_partitions_with_state":
            func = args[1]
            state = args[0]
            args = (state, input, *args[2:])

        if not callable(func):
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
    udf_time = (time.perf_counter_ns() - udf_time_start) // 1000

    # astype can fail in some cases when input is empty
    if len(out):
        # TODO: verify this is correct for all possible result_type's
        if out.dtype != pd.ArrowDtype(result_type):
            out = out.astype(pd.ArrowDtype(result_type))
    else:
        out = pd.Series(_empty_pd_array(result_type), index=out.index, name=out.name)

    if out.name is None:
        out.name = "OUT"

    py_to_cpp_start = time.perf_counter_ns()
    if is_series:
        out_ptr = plan_optimizer.arrow_array_to_cpp_table(
            out.array._pa_array.combine_chunks(), str(out.name), cpp_table
        )
    else:
        out_ptr, _ = df_to_cpp_table(pd.DataFrame({out.name: out}))
    py_to_cpp = (time.perf_counter_ns() - py_to_cpp_start) // 1000
    return out_ptr, cpp_to_py, udf_time, py_to_cpp


def write_s3_vectors_helper(cpp_table, vector_bucket_name, index_name, region):
    """Write a C++ table to S3 Vectors using the boto3 client."""
    import boto3

    df = cpp_table_to_df(cpp_table)

    if not len(df):
        return

    df = df.loc[:, ["key", "data", "metadata"]]
    df["data"] = df.data.map(lambda x: {"float32": x.tolist()})

    s3vectors = boto3.client("s3vectors", region_name=region)
    s3vectors.put_vectors(
        vectorBucketName=vector_bucket_name,
        indexName=index_name,
        vectors=df.to_dict(orient="records"),
    )


def query_s3_vectors_helper(
    S,
    vector_bucket_name,
    index_name,
    region,
    topk,
    filter,
    return_distance,
    return_metadata,
):
    """Query S3 Vectors using the boto3 client."""
    import boto3

    s3vectors = boto3.client("s3vectors", region_name=region)

    keys = []
    distances = []
    metadata = []

    for embedding in S:
        response = s3vectors.query_vectors(
            vectorBucketName=vector_bucket_name,
            indexName=index_name,
            queryVector={"float32": embedding},
            topK=topk,
            filter=filter,
            returnDistance=return_distance,
            returnMetadata=return_metadata,
        )
        vectors = response.get("vectors", [])
        keys.append([v["key"] for v in vectors])
        if return_distance:
            distances.append([v["distance"] for v in vectors])
        if return_metadata:
            metadata.append([str(v["metadata"]) for v in vectors])

    out_keys = pa.array(keys, type=pa.large_list(pa.large_string()))
    arrs = [out_keys]
    names = ["keys"]

    if return_distance:
        out_distances = pa.array(distances, type=pa.large_list(pa.float32()))
        arrs.append(out_distances)
        names.append("distances")

    if return_metadata:
        out_metadata = pa.array(metadata, type=pa.large_list(pa.large_string()))
        arrs.append(out_metadata)
        names.append("metadata")

    out_arr = pa.StructArray.from_arrays(arrs, names=names)
    return pd.Series(out_arr, name=S.name, index=S.index)


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
    from bodo.pandas.plan import LazyPlan
    from bodo.pandas.series import BodoSeries

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


def _is_generated_index_name(name):
    """Check if the Index name is a generated name similar to PyArrow:
    https://github.com/apache/arrow/blob/5e9fce493f21098d616f08034bc233fcc529b3ad/python/pyarrow/pandas_compat.py#L1071
    """
    import re

    pattern = r"^__index_level_\d+__$"
    return re.match(pattern, name) is not None


def _fix_multi_index_names(names: list[str]) -> list[str]:
    """Replace instances of BODO_NONE_DUMMY in MultiIndex names with None
    to ensure missing index names round trip correctly from Arrow."""
    return [None if n == BODO_NONE_DUMMY else n for n in names]


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
            # Index not found in table: matching Pyarrow's behavior, which treats
            # missing index as RangeIndex.
            if descr not in df:
                continue
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
        index_names = _fix_multi_index_names(index_names)
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
        {
            field.name: _empty_pd_array(field.type, field_name=field.name)
            for field in arrow_schema
        }
    )
    return _reconstruct_pandas_index(empty_df, arrow_schema)


def _fix_struct_arr_names(arr, pa_type):
    """Fix the names of the fields in a struct array to match the Arrow type.
    This is necessary since our C++ code may not preserve the field names in
    struct arrays.
    """

    # Handle list recursively
    if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        new_arr = pa.LargeListArray.from_arrays(
            arr.offsets, _fix_struct_arr_names(arr.values, pa_type.value_type)
        )
        # Arrow's from_arrays ignores nulls (bug as of Arrow 13) so we add them back manually
        return pa.Array.from_buffers(
            new_arr.type, len(new_arr), arr.buffers()[:2], children=[new_arr.values]
        )

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


def _arrow_array_to_pd(arrow_array, pa_type, use_arrow_dtypes=True, name=None):
    """Convert a PyArrow array to a pandas array with the specified Arrow type."""

    # Our type inference may fail for some object columns so use the proper Arrow type
    if pa_type == pa.null():
        pa_type = arrow_array.type

    # Our C++ code may not preserve the field names in struct arrays
    # so we fix them here to match the Arrow schema.
    arrow_array = _fix_struct_arr_names(arrow_array, pa_type)

    # Cast to expected type to match Pandas (as determined by the frontend)
    if pa_type != arrow_array.type:
        arrow_array = arrow_array.cast(pa_type)

    if use_arrow_dtypes:
        return pd.Series(
            arrow_array, dtype=pd.ArrowDtype(pa_type), name=name, copy=False
        )

    out = arrow_array.to_pandas()
    if name:
        out.name = name
    return out


def arrow_table_to_pandas(arrow_table, arrow_schema=None, use_arrow_dtypes=True):
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
            i: _arrow_array_to_pd(arrow_table.columns[i], field.type, use_arrow_dtypes)
            for i, field in enumerate(arrow_schema)
        },
        copy=False,
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
        method_name ({"apply", "map", "map_parititons", None}): The name of the method
            applying the UDF. None means it's a function being called directly and not
            a method.
        func (Any): The UDF argument to pass to apply/map.
        kwargs (dict): Optional keyword arguments to pass to apply/map.

    Raises:
        BodoLibNotImplementedException: If the dtype cannot be infered.

    Returns:
        Empty Series with the dtype matching the output of the UDF
        (or equivalent pyarrow dtype)
    """
    assert method_name in {
        "map",
        "apply",
        "map_partitions",
        "map_with_state",
        "map_partitions_with_state",
        None,
    }, (
        "expected method to be one of {'apply', 'map', 'map_partitions', 'map_with_state', 'map_partitions_with_state', None}"
    )

    base_class = obj.__class__.__bases__[0]

    # map_partitions is not a pandas.DataFrame method.
    apply_method = None
    if method_name in ("map", "apply"):
        apply_method = getattr(base_class, method_name)

    # TODO: Tune sample sizes
    sample_sizes = (1, 4, 9, 25, 100)

    except_msg = ""
    for sample_size in sample_sizes:
        pd_sample = base_class(obj.head(sample_size))

        if method_name == "map_with_state":
            out_sample = pd_sample.apply(lambda row: func[1](func[0], row))
        elif method_name == "map_partitions_with_state":
            out_sample = func[1](func[0], pd_sample, *args, **kwargs)
        else:
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


def fallback_warn(msg):
    if bodo.dataframe_library_warn:
        warnings.warn(BodoLibFallbackWarning(msg))


class FallbackContext:
    """Context manager for tracking nested fallback calls."""

    level = 0

    @classmethod
    def is_top_level(cls):
        """Check we are in the top level context i.e. this fallback was not triggered
        by another fallback."""
        return FallbackContext.level == 0

    def __enter__(self):
        FallbackContext.level += 1

    def __exit__(self, exc_type, exc_value, traceback):
        FallbackContext.level -= 1


# TODO: further generalize. Currently, this method is only used for BodoSeries and BodoDataFrame.
def fallback_wrapper(self, attr, name, msg):
    """
    Wrap callable attributes with a warning silencer, unless they are known
    accessors or indexers like `.iloc`, `.loc`, `.str`, `.dt`, `.cat`.
    """

    # Avoid wrapping indexers & accessors
    if (
        callable(attr)
        and not hasattr(attr, "__getitem__")
        and not hasattr(attr, "__getattr__")
    ):

        def silenced_method(*args, **kwargs):
            jit_fallback = JITFallback(self, name)
            try:
                return jit_fallback(*args, **kwargs)
            except Exception:
                pass

            nonlocal msg
            fallback_warn(msg)
            msg = ""
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=BodoLibFallbackWarning)
                try:
                    with FallbackContext():
                        py_res = attr(*args, **kwargs)

                    # Convert objects to Bodo before returning them to the user.
                    if FallbackContext.is_top_level():
                        return convert_to_bodo(py_res)

                    return py_res
                except TypeError as e:
                    msg = e

            # In some cases, fallback fails and raises TypeError due to some operations being unsupported between PyArrow types.
            # Below logic processes deeper fallback that converts problematic PyArrow types to their Pandas equivalents.
            if isinstance(self, bodo.pandas.BodoSeries):
                pd_self = pd.Series(self)

                # When self.dtype is pd.ArrowDtype(pa.timestamp("ns")), apply to_datetime elementwise.
                if isinstance(self.dtype, pd.ArrowDtype) and pa.types.is_timestamp(
                    self.dtype.pyarrow_dtype
                ):
                    fallback_warn(
                        "TypeError triggering deeper fallback. Converting PyarrowDtype elements in self to Pandas dtypes."
                    )
                    converted = pd_self.array._pa_array.to_pandas()
                    return convert_to_bodo(
                        getattr(converted, attr.__name__)(*args[1:], **kwargs)
                    )

            # Raise TypeError from initial call if self does not fall into any of the covered cases.
            raise TypeError(msg)

        return silenced_method

    return attr


def single_arg_check_no_jit(v1, v2):
    """Same as bodo.utils.typing.single_arg_check, but without JIT support to avoid
    JIT import.
    """
    import numpy as np

    return (
        (v1 is not None and v2 is None)
        or (v1 is None and v2 is not None)
        or (v1 is not np.nan and v1 != v2)
        or (v1 is np.nan and v2 is not np.nan)
        or (v1 is not np.nan and v2 is np.nan)
    )


def check_unsupported_args_fallback(
    fname,
    must_be_default_args,
    must_be_default_kwargs,
    args,
    kwargs,
    package_name="pandas",
    fn_str=None,
    module_name="",
    raise_on_error=False,
):
    """Check for unsupported arguments for function 'fname', and raise an error if any
    value other than the default is provided.
    'args_dict' is a dictionary of provided arguments in overload.
    'arg_defaults_dict' is a dictionary of default values for unsupported arguments.

    'package_name' is used to differentiate by various libraries in documentation links (i.e. numpy, pandas)

    'module_name' is used for libraries that are split into multiple different files per module.

    'raise_on_error' to generate exception on unsupported usage else return whether unsupported usage occurred.
    """

    if fn_str == None:
        fn_str = f"{fname}()"
    error_message = ""
    unsupported = False

    # Check all the arguments given positionally that have to have their default values.
    for idx, param in must_be_default_args.items():
        # If parameter index is greater than number of args then nothing left to check.
        if idx >= len(args):
            break
        v1 = args[idx]  # Get the actual value.
        v2 = param.default  # Get the default value.
        # Flexible check for not matching.
        if single_arg_check_no_jit(v1, v2):
            error_message = (
                f"{fn_str}: {param.name} parameter only supports default value {v2}"
            )
            unsupported = True
            break

    # Check all the keyword arguments that have to have their default values if we
    # haven't already found an error.
    if not unsupported:
        for name, param in must_be_default_kwargs.items():
            if name not in kwargs:
                continue
            v1 = kwargs[name]  # Get the actual value.
            v2 = param.default  # Get the default value.
            # Flexible check for not matching.
            if single_arg_check_no_jit(v1, v2):
                error_message = (
                    f"{fn_str}: {name} parameter only supports default value {v2}"
                )
                unsupported = True
                break

    if not raise_on_error:
        return unsupported

    raise ValueError(error_message)


class JITFallback:
    # Holds a mapping of a tuple of class name and function name to either
    # False to say that compilation previously failed for that function or
    # a callable dispatcher for the JIT compiled version of that function.
    fallback_cache = {}
    compile_success = 0
    compile_fail = 0
    after_success = 0
    python_fallback = 0

    class JITFallbackFail(Exception):
        pass

    def __init__(self, base_obj, name):
        self.base_obj = base_obj if base_obj is not pd else None
        self.name = name

    def __call__(self, *args, **kwargs):
        import bodo

        key = (
            (
                self.base_obj.__class__.__name__,
                self.name,
                *[type(x) for x in args],
                *list(kwargs.keys()),
                *[type(x) for x in kwargs.values()],
            ),
            None,
        )
        # See if we previously tried to compile this function.
        cache_entry = JITFallback.fallback_cache.get(key, None)
        # We are using an allow-list strategy below because many of the possible
        # functions return data-types that don't have equivalents in this
        # dataframe library yet or we don't have conversions from the Python
        # type to the corresponding dataframe library type or plain just don't
        # work for unknown reasons.  So, we will be on the safe side for now and
        # only JIT fallback for methods that we have tested.
        if self.name in ("duplicated", "pivot") and cache_entry != False:
            # Import compiler
            import bodo.decorators  # isort:skip # noqa

            bodo.spawn.utils.import_compiler_on_workers()

            # None means it wasn't in the cache either way so we can try to
            # JIT compile it.
            if cache_entry is None:
                if self.base_obj is None:
                    from bodo.numba_compat import is_func_overloaded

                    # Do a better check here if this function appears overloaded.
                    jit_supported = is_func_overloaded("pandas", self.name)
                else:
                    overloads = get_overloads(self.base_obj.__class__.__name__)
                    jit_supported = self.name in overloads

                if jit_supported:
                    fname = f"bodo_jitfallback_{self.name}"
                    self_arg = "self, " if self.base_obj is not None else ""
                    sig_args = ",".join(
                        [f"arg{i}" for i in range(len(args))] + list(kwargs.keys())
                    )
                    caller_args = ",".join(
                        [f"arg{i}" for i in range(len(args))]
                        + [f"{x}={x}" for x in kwargs.keys()]
                    )
                    func_text = f"def {fname}({self_arg}{sig_args}):\n"
                    if self.base_obj is None:
                        func_text += f"    return pd.{self.name}({caller_args})\n"
                    else:
                        func_text += f"    return self.{self.name}({caller_args})\n"

                    new_func = bodo.utils.utils.bodo_spawn_exec(
                        func_text, {"pd": pd}, {}, __name__
                    )
                    compiled_method = bodo.jit(new_func, cache=True)
                    try:
                        if self.base_obj is None:
                            cm_args = args + tuple(kwargs.values())
                        else:
                            cm_args = (self.base_obj, *args, *tuple(kwargs.values()))
                        ret = compiled_method(*cm_args)
                        # Remember that this compile worked.
                        JITFallback.fallback_cache[key] = compiled_method
                        JITFallback.compile_success += 1
                        return ret
                    except Exception:
                        # Remember not to try to compile this again.
                        JITFallback.fallback_cache[key] = False
                        JITFallback.compile_fail += 1
                else:
                    JITFallback.fallback_cache[key] = False
            else:
                JITFallback.after_success += 1
                # Previous successful compile so just run it.
                if self.base_obj is None:
                    cm_args = args + tuple(kwargs.values())
                else:
                    cm_args = (self.base_obj, *args, *tuple(kwargs.values()))
                return cache_entry(*cm_args)

        JITFallback.python_fallback += 1
        raise JITFallback.JITFallbackFail()


def insert_bodo_scalar(
    plan: LazyPlan, scalar: BodoScalar
) -> tuple[LazyPlan, ColRefExpression]:
    """
    Insert a scalar as a column in the given plan.
    Args:
        plan (LazyPlan): The plan to insert the scalar into.
        scalar (BodoScalar): The scalar to insert.
    Returns:
        Tuple[LazyPlan, ColRefExpression]: The new plan with the scalar inserted as a column,
            and the column reference expression for the new column.
    """
    from bodo.pandas.base import _empty_like
    from bodo.pandas.plan import (
        ColRefExpression,
        LogicalInsertScalarSubquery,
        LogicalProjection,
    )

    assert scalar.is_lazy_plan(), (
        "Expected scalar to have a lazy plan, use a constant expression if the scalar is not lazy."
    )

    empty_data = plan.empty_data.copy()
    col_name = "_scalar_col"
    n_indices = get_n_index_arrays(empty_data.index)
    n_orig_data_cols = len(empty_data.columns)

    if n_indices > 0:
        # Cross join adds left index after left columns, then the scalar column.
        # Therefore, move Index columns to the end before adding new scalar column.
        # E.g. [A, B, C, index, _scalar_col]
        empty_data = empty_data.reset_index()
        index_cols = empty_data.columns[:n_indices]
        data_cols = empty_data.columns[n_indices:]
        empty_data = empty_data[list(data_cols) + list(index_cols)]

    empty_data[col_name] = _empty_like(scalar.wrapped_series)

    new_plan = LogicalInsertScalarSubquery(empty_data, plan, scalar._plan)

    if n_indices > 0:
        # Move scalar column last right before index columns as expected in rest of
        # the code.
        # E.g. [A, B, C, _scalar_col, index]
        exprs = [
            ColRefExpression(empty_data.iloc[:, i].to_frame(), new_plan, i)
            for i in range(n_orig_data_cols)
        ]
        scalar_col = empty_data.shape[1] - 1
        exprs.append(
            ColRefExpression(
                empty_data.iloc[:, scalar_col].to_frame(), new_plan, scalar_col
            )
        )
        exprs += [
            ColRefExpression(empty_data.iloc[:, i].to_frame(), new_plan, i)
            for i in range(n_orig_data_cols, n_orig_data_cols + n_indices)
        ]

        index_cols = [
            empty_data.columns[i]
            for i in range(n_orig_data_cols, n_orig_data_cols + n_indices)
        ]
        empty_data = empty_data.set_index(index_cols)
        new_plan = LogicalProjection(empty_data, new_plan, exprs)

    col_expr = ColRefExpression(
        empty_data[col_name].to_frame(), new_plan, empty_data.shape[1] - 1
    )
    return new_plan, col_expr


def log_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Inspect the call stack: frame 0 = wrapper, 1 = func, 2 = caller
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        from bodo.pandas import BodoDataFrame, BodoSeries

        log_entry = True
        while caller_frame:
            caller_module = caller_frame.f_globals.get("__name__", "")
            if caller_module.startswith("bodo"):
                log_entry = False
                break
            caller_frame = caller_frame.f_back

        # Only log if the caller is *not* in the bodo.* hierarchy
        if log_entry:

            def log_repr(x):
                if x.__class__.__name__ == "BodoDataFrame":
                    return f"df{id(x)}"
                elif x.__class__.__name__ == "BodoSeries":
                    return f"s{id(x)}"
                else:
                    return f"{type(x).__name__}({repr(x)})"

            arg_str = ", ".join(
                [
                    *[log_repr(a) for a in args],
                    *[f"{k}={log_repr(v)}" for k, v in kwargs.items()],
                ]
            )
            call_str = f"{func.__module__}.{func.__qualname__}({arg_str})\n"

        ret = func(*args, **kwargs)
        if log_entry:
            if isinstance(ret, BodoDataFrame):
                call_str = f"df{id(ret)} = {call_str}"
            elif isinstance(ret, BodoSeries):
                call_str = f"s{id(ret)} = {call_str}"
            with open("bodo.capture", "a") as _log_file:
                _log_file.write(call_str)
                _log_file.flush()
        return ret

    return wrapper


def wrap_module_functions_and_methods(module):
    if not bodo.dataframe_library_capture:
        return
    with open("bodo.capture", "a") as _log_file:
        for name, obj in vars(module).items():
            # Wrap top-level functions
            if (
                isinstance(obj, pytypes.FunctionType)
                and obj.__module__ == module.__name__
            ):
                setattr(module, name, log_wrapper(obj))

            # Wrap methods in classes
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                for attr_name, attr in vars(obj).items():
                    if isinstance(
                        attr, (pytypes.FunctionType, classmethod, staticmethod)
                    ):
                        original = attr
                        if isinstance(attr, (classmethod, staticmethod)):
                            original = attr.__func__
                        wrapped = log_wrapper(original)
                        if isinstance(attr, classmethod):
                            wrapped = classmethod(wrapped)
                        elif isinstance(attr, staticmethod):
                            wrapped = staticmethod(wrapped)
                        setattr(obj, attr_name, wrapped)


def scalarOutputNACheck(out, dtype):
    """Pandas will convert some types to float and return NaN
    if there is no data.
    """
    if isinstance(out, pd._libs.missing.NAType):
        if isinstance(dtype, pd.ArrowDtype):
            dtype = dtype.numpy_dtype

        if np.issubdtype(dtype, np.floating):
            return np.nan
        elif np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
            # plain NumPy ints/bools can't hold NA, pandas promotes to float NaN
            return np.nan
    return out
