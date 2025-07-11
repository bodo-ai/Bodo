from __future__ import annotations

import functools
import importlib
import inspect
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


def cpp_table_to_df(cpp_table, arrow_schema=None, use_arrow_dtypes=True):
    """Convert a C++ table (table_info) to a pandas DataFrame."""
    from bodo.ext import plan_optimizer

    arrow_table = plan_optimizer.cpp_table_to_arrow(cpp_table)
    df = arrow_table_to_pandas(arrow_table, arrow_schema, use_arrow_dtypes)
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
def get_series_datetime_overloads():
    """Return a list of the functions supported on SeriesDatetimePropertiesType objects
    to some degree by bodo.jit.
    """
    from bodo.hiframes.series_dt_impl import SeriesDatetimePropertiesType
    from bodo.numba_compat import get_method_overloads

    return get_method_overloads(SeriesDatetimePropertiesType)


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
                    elif self.__class__ == bodo.pandas.series.BodoDatetimeProperties:
                        base_class = self._series.__class__.__bases__[0].dt
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
                    py_res = fallback_wrapper(getattr(base_class, func.__name__))(
                        self, *args, **kwargs
                    )
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
    func, is_series, is_attr, args, kwargs = in_args

    # Arrow dtypes can be very slow for UDFs in Pandas:
    # https://github.com/pandas-dev/pandas/issues/61747
    # TODO[BSE-4948]: Use Arrow dtypes when Bodo engine is specified
    use_arrow_dtypes = not (is_attr and func == "apply")
    input = cpp_table_to_df(cpp_table, use_arrow_dtypes=use_arrow_dtypes)

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


def _arrow_to_pd_array(arrow_array, pa_type, use_arrow_dtypes=True):
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

    if use_arrow_dtypes:
        return pd.array(arrow_array, dtype=pd.ArrowDtype(pa_type))

    return arrow_array.to_pandas()


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
            i: _arrow_to_pd_array(arrow_table.columns[i], field.type, use_arrow_dtypes)
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


def fallback_wrapper(attr):
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=BodoLibFallbackWarning)
                return attr(*args, **kwargs)

        return silenced_method

    return attr
