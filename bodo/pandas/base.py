"""Support for top level pandas functions.
This file should import JIT lazily to avoid slowing down non-JIT code paths.
"""

from __future__ import annotations

import csv
import importlib
import sys
import typing as pt
import warnings
from collections.abc import (
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)

import pandas as pd
import pyarrow as pa
from pandas._libs import lib
from pandas._typing import (
    Axis,
    CompressionOptions,
    CSVEngine,
    DtypeArg,
    DtypeBackend,
    FilePath,
    Hashable,
    HashableT,
    IndexLabel,
    ReadCsvBuffer,
    StorageOptions,
    UsecolsArgType,
)
from pandas.core.tools.datetimes import _unit_map
from pandas.io.parsers.readers import _c_parser_defaults

import bodo.spawn.spawner  # noqa: F401
from bodo.io.parquet_pio import get_parquet_dataset, parquet_dataset_unify_nulls
from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.plan import (
    LazyPlanDistributedArg,
    LogicalGetIcebergRead,
    LogicalGetPandasReadParallel,
    LogicalGetPandasReadSeq,
    LogicalGetParquetRead,
    LogicalLimit,
    LogicalProjection,
    LogicalSetOperation,
    NullExpression,
    _get_df_python_func_plan,
    make_col_ref_exprs,
)
from bodo.pandas.scalar import BodoScalar
from bodo.pandas.series import BodoSeries, _get_series_func_plan
from bodo.pandas.utils import (
    BODO_NONE_DUMMY,
    BodoDictionaryTypeInvalidException,
    BodoLibNotImplementedException,
    arrow_to_empty_df,
    check_args_fallback,
    ensure_datetime64ns,
    get_scalar_udf_result_type,
    wrap_module_functions_and_methods,
    wrap_plan,
)

if pt.TYPE_CHECKING:
    from pyiceberg.table import Table as PyIcebergTable


def from_pandas(df):
    """Convert a Pandas DataFrame to a BodoDataFrame."""

    import bodo

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if isinstance(df.columns, pd.MultiIndex):
        raise BodoLibNotImplementedException(
            "from_pandas(): Hierarchical column names are not supported in Bodo yet."
        )

    if df.columns.has_duplicates:
        raise BodoLibNotImplementedException(
            "from_pandas(): Duplicate column names are not supported in Bodo yet."
        )

    new_columns = []
    for c in df.columns:
        if not isinstance(c, str):
            warnings.warn(
                f"The column name '{c}' with type {type(c)} was converted to string."
            )
        new_columns.append(str(c))

    df.columns = new_columns

    # Avoid datetime64[us] that is commonly used in Pandas but not supported in Bodo.
    df = ensure_datetime64ns(df)

    # Make sure empty_df has proper dtypes since used in the plan output schema.
    # Using sampling to avoid large memory usage.
    sample_size = 100

    # TODO [BSE-4788]: Refactor with convert_to_arrow_dtypes util
    for col in df.select_dtypes(include=["object"]).columns:
        if len(df[col]) > 0 and type(df[col].iloc[0]) is BodoScalar:
            df[col] = df[col].apply(lambda x: x.get_value() if x is not None else None)

    try:
        pa_schema = pa.Schema.from_pandas(df.iloc[:sample_size])
    except pa.lib.ArrowInvalid as e:
        # TODO: add specific unsupported columns to message.
        raise BodoLibNotImplementedException(
            "from_pandas(): Could not convert DataFrame to Bodo: "
            + "Unsupported datatype encountered in one or more columns: "
            + str(e)
        )

    try:
        empty_df = arrow_to_empty_df(pa_schema)
    except BodoDictionaryTypeInvalidException as e:
        raise BodoLibNotImplementedException(
            "from_pandas(): Could not convert DataFrame to Bodo: " + str(e)
        )

    n_rows = len(df)

    res_id = None
    if bodo.dataframe_library_run_parallel:
        plan = LogicalGetPandasReadParallel(
            empty_df,
            n_rows,
            LazyPlanDistributedArg(df),
        )
    else:
        plan = LogicalGetPandasReadSeq(empty_df, df)

    return wrap_plan(plan=plan, nrows=n_rows, res_id=res_id)


@check_args_fallback(supported=["dtype_backend"])
def read_parquet(
    path,
    engine="auto",
    columns=None,
    storage_options=None,
    use_nullable_dtypes=lib.no_default,
    dtype_backend=lib.no_default,
    filesystem=None,
    filters=None,
    **kwargs,
):
    if dtype_backend is not lib.no_default and dtype_backend != "pyarrow":
        raise BodoLibNotImplementedException(
            "read_parquet() only supports dtype_backend='pyarrow' in Bodo."
        )

    if storage_options is None:
        storage_options = {}

    # Read Parquet schema
    use_hive = True
    pq_dataset = get_parquet_dataset(
        path,
        get_row_counts=False,
        storage_options=storage_options,
        partitioning="hive" if use_hive else None,
    )
    pq_dataset = parquet_dataset_unify_nulls(pq_dataset)
    arrow_schema = pq_dataset.schema
    # Convert dictionary columns to use int32 indices since our c++ dict implementation
    # only supports int32 indices.
    for i in range(len(arrow_schema)):
        field = arrow_schema.field(i)
        if pa.types.is_dictionary(field.type):
            arrow_schema = arrow_schema.set(
                i,
                pa.field(
                    field.name,
                    pa.dictionary(
                        pa.int32(), field.type.value_type, field.type.ordered
                    ),
                    field.nullable,
                    field.metadata,
                ),
            )

    empty_df = arrow_to_empty_df(arrow_schema)

    plan = LogicalGetParquetRead(
        empty_df,
        path,
        storage_options,
    )
    return wrap_plan(plan=plan)


def merge(lhs, rhs, *args, **kwargs):
    return lhs.merge(rhs, *args, **kwargs)


def _empty_like(val):
    """Create an empty Pandas DataFrame or Series having the same schema as
    the given BodoDataFrame or BodoSeries or Pandas DataFrame or Series.
    For Pandas DataFrame or Series, uses Arrow for schema inference of object columns
    and returns typed output.
    """
    import numpy as np
    import pyarrow as pa

    if type(val) not in (
        BodoDataFrame,
        BodoSeries,
        BodoScalar,
        pd.Series,
        pd.DataFrame,
    ):
        raise TypeError(f"val must be a BodoDataFrame or BodoSeries, got {type(val)}")

    if type(val) is BodoScalar:
        return val.wrapped_series.head(0).dtype.type()

    is_series = isinstance(val, (BodoSeries, pd.Series))

    if isinstance(val, (BodoDataFrame, BodoSeries)):
        # Avoid triggering data collection
        # Ok since BodoDataFrame/Series always have Arrow schema and not objects
        val = val.head(0)

    if is_series:
        val = val.to_frame(name=BODO_NONE_DUMMY if val.name is None else val.name)

    # Work around categorical gaps in Arrow-Pandas conversion
    original_val = val
    cat_cols = set()
    for cname, dtype in val.dtypes.items():
        if isinstance(dtype, pd.CategoricalDtype):
            cat_cols.add(cname)
            val = val.assign(**{cname: np.arange(len(val))})

    is_cat_index = isinstance(val.index, pd.CategoricalIndex)
    if is_cat_index:
        val = val.reset_index(drop=True)

    # Reuse arrow_to_empty_df to make sure details like Index handling are correct
    out = arrow_to_empty_df(pa.Schema.from_pandas(val))

    for cname in cat_cols:
        out[cname] = original_val[cname].iloc[:0]

    if is_cat_index:
        out.index = original_val.index[:0]

    if isinstance(original_val.index, (pd.PeriodIndex, pd.IntervalIndex)):
        out.index = original_val.index[:0]

    if is_series:
        out = out.iloc[:, 0]

    return out


@check_args_fallback(
    supported=[
        "catalog_name",
        "catalog_properties",
        "selected_fields",
        "limit",
        "row_filter",
        "snapshot_id",
        "location",
    ]
)
def read_iceberg(
    table_identifier: str,
    catalog_name: str | None = None,
    *,
    catalog_properties: dict[str, pt.Any] | None = None,
    row_filter: str | None = None,
    selected_fields: tuple[str] | None = None,
    case_sensitive: bool = True,
    snapshot_id: int | None = None,
    limit: int | None = None,
    scan_properties: dict[str, pt.Any] | None = None,
    location: str | None = None,
) -> BodoDataFrame:
    import pyiceberg.catalog
    import pyiceberg.expressions
    import pyiceberg.table

    from bodo.io.iceberg.read_metadata import get_table_length
    from bodo.pandas.utils import BodoLibNotImplementedException

    # Support simple directory only calls like:
    # pd.read_iceberg("table", location="/path/to/table")
    if catalog_name is None and catalog_properties is None and location is not None:
        if location.startswith("arn:aws:s3tables:"):
            from bodo.io.iceberg.catalog.s3_tables import (
                construct_catalog_properties as construct_s3_tables_catalog_properties,
            )

            catalog_properties = construct_s3_tables_catalog_properties(location)
        else:
            catalog_properties = {
                pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
                pyiceberg.catalog.WAREHOUSE_LOCATION: location,
            }
    elif location is not None:
        raise BodoLibNotImplementedException(
            "'location' is only supported for filesystem catalog and cannot be used "
            "with catalog_name or catalog_properties."
        )
    elif catalog_properties is None:
        catalog_properties = {}

    catalog = pyiceberg.catalog.load_catalog(catalog_name, **catalog_properties)

    # Get the output schema
    table = catalog.load_table(table_identifier)
    pyiceberg_schema = table.schema()
    arrow_schema = pyiceberg_schema.as_arrow()
    empty_df = arrow_to_empty_df(arrow_schema)

    # Get the table length estimate, if there's not a filter it will be exact
    table_len_estimate = get_table_length(table, snapshot_id or -1)

    # If there's a row filter, we need to estimate the selectivity
    # and adjust the table length estimate accordingly.
    if row_filter is not None and table_len_estimate > 0:
        # TODO: do something smarter here like sampling or turn the filter into a
        # separate node so the planner can handle it
        #
        # This matches duckdb's default selectivity estimate for filters
        filter_selectivity_estimate = 0.2
        table_len_estimate = int(table_len_estimate * filter_selectivity_estimate)

    plan = LogicalGetIcebergRead(
        empty_df,
        table_identifier,
        catalog_name,
        catalog_properties,
        pyiceberg.table._parse_row_filter(row_filter)
        if row_filter
        else pyiceberg.expressions.AlwaysTrue(),
        # We need to pass the pyiceberg schema so we can bind the iceberg filter to it
        # during filter conversion. See bodo/io/iceberg/common.py::pyiceberg_filter_to_pyarrow_format_str_and_scalars
        pyiceberg_schema,
        snapshot_id if snapshot_id is not None else -1,
        table_len_estimate,
        arrow_schema=arrow_schema,
    )

    if selected_fields is not None:
        col_idxs = {
            arrow_schema.get_field_index(field_name) for field_name in selected_fields
        }
        empty_df = empty_df[list(selected_fields)]
    else:
        # Adds logical projection layer to enable rename.
        col_idxs = range(len(empty_df.columns))

    exprs = make_col_ref_exprs(col_idxs, plan)
    plan = LogicalProjection(
        empty_df,
        plan,
        exprs,
    )

    if limit is not None:
        plan = LogicalLimit(
            empty_df,
            plan,
            limit,
        )

    return wrap_plan(plan=plan)


def read_iceberg_table(table: PyIcebergTable) -> BodoDataFrame:
    import pyiceberg.catalog

    # We can't scatter catalogs so we need to use properties instead so the workers can
    # create the catalog themselves.
    catalog_properties = table.catalog.properties

    # NOTE: catalog implementation and type cannot be set at the same time:
    # https://github.com/ehsantn/iceberg-python/blob/cae24259aa7ea3923703f65b58da7ff5a67414ba/pyiceberg/catalog/__init__.py#L242
    if pyiceberg.catalog.TYPE not in catalog_properties:
        catalog_properties[pyiceberg.catalog.PY_CATALOG_IMPL] = (
            table.catalog.__class__.__module__ + "." + table.catalog.__class__.__name__
        )

    return read_iceberg(
        ".".join(table._identifier),
        catalog_name=table.catalog.name,
        catalog_properties=catalog_properties,
    )


@check_args_fallback(
    supported=[
        "names",
        "usecols",
        "parse_dates",
    ]
)
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = lib.no_default,
    delimiter: str | None | lib.NoDefault = None,
    # Column and Index Locations and Names
    header: int | Sequence[int] | None | pt.Literal["infer"] = "infer",
    names: Sequence[Hashable] | None | lib.NoDefault = lib.no_default,
    index_col: IndexLabel | pt.Literal[False] | None = None,
    usecols: UsecolsArgType = None,
    # General Parsing Configuration
    dtype: DtypeArg | None = None,
    engine: CSVEngine | None = None,
    converters: Mapping[Hashable, pt.Callable] | None = None,
    true_values: list | None = None,
    false_values: list | None = None,
    skipinitialspace: bool = False,
    skiprows: list[int] | int | pt.Callable[[Hashable], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    # NA and Missing Data Handling
    na_values: Hashable
    | Iterable[Hashable]
    | Mapping[Hashable, Iterable[Hashable]]
    | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool | lib.NoDefault = lib.no_default,
    skip_blank_lines: bool = True,
    # Datetime Handling
    parse_dates: bool | Sequence[Hashable] | None = None,
    infer_datetime_format: bool | lib.NoDefault = lib.no_default,
    keep_date_col: bool | lib.NoDefault = lib.no_default,
    date_parser: pt.Callable | lib.NoDefault = lib.no_default,
    date_format: str | dict[Hashable, str] | None = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    # Iteration
    iterator: bool = False,
    chunksize: int | None = None,
    # Quoting, Compression, and File Format
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: int = csv.QUOTE_MINIMAL,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    # Error Handling
    on_bad_lines: str = "error",
    # Internal
    delim_whitespace: bool | lib.NoDefault = lib.no_default,
    low_memory: bool = _c_parser_defaults["low_memory"],
    memory_map: bool = False,
    float_precision: pt.Literal["high", "legacy"] | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> BodoDataFrame:
    # Import compiler
    import bodo.decorators  # isort:skip # noqa
    from bodo.utils.utils import bodo_spawn_exec

    bodo.spawn.utils.import_compiler_on_workers()

    func = "def bodo_read_csv(filepath"
    if names != lib.no_default:
        func += ", names"
    if usecols != None:
        func += ", usecols"
    if parse_dates != None:
        func += ", parse_dates"
    func += "):\n"
    func += "    return pd.read_csv(filepath"
    func_args = []
    if names != lib.no_default:
        func += ", names=names"
        func_args.append(names)
    if usecols != None:
        func += ", usecols=usecols"
        func_args.append(usecols)
    if parse_dates != None:
        func += ", parse_dates=parse_dates"
        func_args.append(parse_dates)
    func += ")\n"
    csv_func = bodo_spawn_exec(func, {"pd": pd}, {}, __name__)
    jit_csv_func = bodo.jit(csv_func, cache=True)
    return jit_csv_func(filepath_or_buffer, *func_args)


def _is_not_tz_format(format: str) -> bool:
    """Check if the given datetime format string does not contain timezone info."""
    tz_indicators = ["%z", "%Z", "%:z", "%::z", "%:::z"]
    for indicator in tz_indicators:
        if indicator in format:
            return False
    return True


@check_args_fallback("none")
def to_datetime(
    arg,
    errors="raise",
    dayfirst=False,
    yearfirst=False,
    utc=False,
    format=None,
    exact=lib.no_default,
    unit=None,
    origin="unix",
    cache=True,
):
    """
    Converts elements of a BodoSeries to timestamp[ns] type.
    Currently, Bodo only supports arg of either BodoSeries or BodoDataFrame instance, falling back to Pandas otherwise.
    """
    if not isinstance(arg, (BodoSeries, BodoDataFrame)):
        raise BodoLibNotImplementedException(
            "to_datetime() is not supported for arg that is not an instance of BodoSeries or BodoDataFrame. Falling back to Pandas."
        )

    in_kwargs = {
        "errors": errors,
        "dayfirst": dayfirst,
        "yearfirst": yearfirst,
        "utc": utc,
        "format": format,
        "exact": exact,
        "unit": unit,
        "origin": origin,
        "cache": cache,
    }

    name = arg.name if isinstance(arg, BodoSeries) else None

    if utc:
        dtype = pd.ArrowDtype(pa.timestamp("ns", tz="UTC"))
        index = arg.head(0).index
        new_metadata = pd.Series(
            dtype=dtype,
            index=index,
            name=name,
        )
    # Format specified without timezone info or DataFrame case (cannot have timezone)
    elif (format is not None and _is_not_tz_format(format)) or isinstance(
        arg, BodoDataFrame
    ):
        dtype = pd.ArrowDtype(pa.timestamp("ns"))
        index = arg.head(0).index
        new_metadata = pd.Series(
            dtype=dtype,
            index=index,
            name=name,
        )
    else:
        # Need to sample the data for output type inference similar to UDFs since the data
        # can have different timezones.
        new_metadata = get_scalar_udf_result_type(
            arg, None, pd.to_datetime, **in_kwargs
        )

    # Avoid using Arrow dtypes for non-timestamp inputs for better performance.
    use_arrow_dtypes = isinstance(arg, BodoSeries) and pa.types.is_timestamp(
        arg.head(0).dtype.pyarrow_dtype
    )

    # 1. DataFrame Case
    if isinstance(arg, BodoDataFrame):
        _validate_df_to_datetime(arg)
        return _get_df_python_func_plan(
            arg._plan,
            new_metadata,
            "pandas.to_datetime",
            (),
            in_kwargs,
            is_method=False,
            use_arrow_dtypes=use_arrow_dtypes,
        )

    # 2. Series Case
    return _get_series_func_plan(
        arg._plan,
        new_metadata,
        "pandas.to_datetime",
        (),
        in_kwargs,
        is_method=False,
        use_arrow_dtypes=use_arrow_dtypes,
    )


@check_args_fallback(unsupported="all")
def concat(
    objs: Iterable[BodoSeries | BodoDataFrame]
    | Mapping[HashableT, BodoSeries | BodoDataFrame],
    *,
    axis: Axis = 0,
    join: str = "outer",
    ignore_index: bool = False,
    keys: Iterable[Hashable] | None = None,
    levels=None,
    names: list[HashableT] | None = None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: bool | None = None,
) -> BodoDataFrame | BodoSeries:
    if isinstance(objs, Mapping):
        raise BodoLibNotImplementedException(
            "concat does not current support objs of Mapping type"
        )

    if len(objs) == 0:
        raise ValueError("No objects to concatenate")
    elif len(objs) == 1:
        return objs[0]

    def concat_two(a, b):
        """Process two dataframes or series to concat just two together."""
        zero_size_a = _empty_like(a)
        zero_size_b = _empty_like(b)

        # Simulate operation in Pandas with empty entities.
        empty_data = pd.concat(
            [zero_size_a, zero_size_b],
            axis=axis,
            join=join,
            ignore_index=ignore_index,
            keys=keys,
            levels=levels,
            names=names,
            sort=sort,
            copy=copy,
        )

        if isinstance(empty_data, pd.DataFrame):

            def get_mapping(new_schema, old_schema, plan):
                """Create col ref expressions to do the reordering between
                the old schema column order and the new one.
                """
                exprs = []
                for field_idx, x in enumerate(new_schema):
                    if x in old_schema:
                        exprs.extend(make_col_ref_exprs([old_schema.index(x)], plan))
                    else:
                        exprs.append(NullExpression(new_schema, plan, field_idx))
                return exprs

            # Create a reordering of the temp a_new_cols so that the columns are in
            # the same order as the Pandas simulation on empty data.
            a_plan = LogicalProjection(
                empty_data,
                a._plan,
                get_mapping(empty_data, a.columns.tolist(), a._plan),
            )
            # Create a reordering of the temp b_new_cols so that the columns are in
            # the same order as the Pandas simulation on empty data.
            b_plan = LogicalProjection(
                empty_data,
                b._plan,
                get_mapping(empty_data, b.columns.tolist(), b._plan),
            )
        else:
            a_plan = a._plan
            b_plan = b._plan

        # DuckDB Union operator requires schema to already be matching.
        # Reverse the order of operands to be more likely to match Pandas output order
        # in SQL.
        planUnion = LogicalSetOperation(empty_data, b_plan, a_plan, "union all")

        return wrap_plan(planUnion)

    # High-level approach is to process two dataframes or series at a time.  If
    # the programmer gave more than 2 then combine the 3rd with the result of
    # first two, the 4th with the result of that and so on.
    cur_res = concat_two(objs[0], objs[1])
    for i in range(2, len(objs)):
        cur_res = concat_two(cur_res, objs[i])

    return cur_res


def _validate_df_to_datetime(df):
    """Validates input dataframe in to_datetime() has correct column names."""
    columns = df._plan.empty_data.columns

    if not columns.is_unique:
        raise ValueError("cannot assemble with duplicate keys")

    def f(value):
        if value in _unit_map:
            return _unit_map[value]

        # m is case significant
        if value.lower() in _unit_map:
            return _unit_map[value.lower()]

        return value

    unit = {k: f(k) for k in df.keys()}
    unit_rev = {v: k for k, v in unit.items()}

    required = ["year", "month", "day"]
    req = sorted(set(required) - set(unit_rev.keys()))
    if len(req):
        _required = ",".join(req)
        raise ValueError(
            "to assemble mappings requires at least that "
            f"[year, month, day] be specified: [{_required}] is missing"
        )

    # keys we don't recognize
    excess = sorted(set(unit_rev.keys()) - set(_unit_map.values()))
    if len(excess):
        _excess = ",".join(excess)
        raise ValueError(
            f"extra keys have been passed to the datetime assemblage: [{_excess}]"
        )


def gen_redirect(name):
    """Returns top-level bodo.pandas redirect method of given name."""

    def _redirect(obj, *args, **kwargs):
        if not isinstance(obj, BodoSeries):
            # If obj is a scalar value, fallback without warning.
            if pd.api.types.is_scalar(obj):
                py_pkg = importlib.import_module("pandas")
                return getattr(py_pkg, name)(obj, *args, **kwargs)
            # TODO: Support isnull, etc. in BodoDataFrame
            raise BodoLibNotImplementedException(
                f"Only supports BodoSeries obj: falling back to Pandas in {name}()"
            )
        func = getattr(BodoSeries, name)
        return func(obj, *args, **kwargs)

    return _redirect


def _install_top_level_redirect():
    """Install bodo.pandas.<method> with redirect."""
    import sys

    for name in ["isna", "isnull", "notna", "notnull"]:
        method = gen_redirect(name)
        method.__name__ = name
        method.__qualname__ = name
        decorated = check_args_fallback("none")(method)
        setattr(sys.modules[__name__], name, decorated)


_install_top_level_redirect()

wrap_module_functions_and_methods(sys.modules[__name__])
