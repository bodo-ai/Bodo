import importlib
import typing as pt

import pandas as pd
import pyarrow as pa
from pandas._libs import lib
from pandas.core.tools.datetimes import _unit_map

from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.series import BodoSeries, _get_series_python_func_plan
from bodo.pandas.utils import (
    BODO_NONE_DUMMY,
    BodoLibNotImplementedException,
    LazyPlan,
    LazyPlanDistributedArg,
    _get_df_python_func_plan,
    arrow_to_empty_df,
    check_args_fallback,
    ensure_datetime64ns,
    make_col_ref_exprs,
    wrap_plan,
)


def from_pandas(df):
    """Convert a Pandas DataFrame to a BodoDataFrame."""

    import bodo

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Avoid datetime64[us] that is commonly used in Pandas but not supported in Bodo.
    df = ensure_datetime64ns(df)

    # Make sure empty_df has proper dtypes since used in the plan output schema.
    # Using sampling to avoid large memory usage.
    sample_size = 100

    # TODO [BSE-4788]: Refactor with convert_to_arrow_dtypes util
    pa_schema = pa.Schema.from_pandas(df.iloc[:sample_size])
    empty_df = arrow_to_empty_df(pa_schema)
    n_rows = len(df)

    res_id = None
    if bodo.dataframe_library_run_parallel:
        nrows = len(df)
        res_id = bodo.spawn.utils.scatter_data(df)
        plan = LazyPlan(
            "LogicalGetPandasReadParallel",
            empty_df,
            nrows,
            LazyPlanDistributedArg(None, res_id),
        )
    else:
        plan = LazyPlan("LogicalGetPandasReadSeq", empty_df, df)

    return wrap_plan(plan=plan, nrows=n_rows, res_id=res_id)


@check_args_fallback("all")
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
    from bodo.io.parquet_pio import get_dataset_unify_nulls

    if storage_options is None:
        storage_options = {}

    # Read Parquet schema
    use_hive = True
    pq_dataset = get_dataset_unify_nulls(
        path,
        storage_options,
        "hive" if use_hive else None,
    )
    arrow_schema = pq_dataset.schema

    empty_df = arrow_to_empty_df(arrow_schema)

    plan = LazyPlan(
        "LogicalGetParquetRead",
        empty_df,
        path,
        storage_options,
    )
    return wrap_plan(plan=plan)


def merge(lhs, rhs, *args, **kwargs):
    return lhs.merge(rhs, *args, **kwargs)


def _empty_like(val):
    """Create an empty Pandas DataFrame or Series having the same schema as
    the given BodoDataFrame or BodoSeries
    """
    import pyarrow as pa

    if not isinstance(val, (BodoDataFrame, BodoSeries)):
        raise TypeError(f"val must be a BodoDataFrame or BodoSeries, got {type(val)}")

    is_series = isinstance(val, BodoSeries)
    # Avoid triggering data collection
    val = val.head(0)

    if is_series:
        val = val.to_frame(name=BODO_NONE_DUMMY if val.name is None else val.name)

    # Reuse arrow_to_empty_df to make sure details like Index handling are correct
    out = arrow_to_empty_df(pa.Schema.from_pandas(val))

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

    plan = LazyPlan(
        "LogicalGetIcebergRead",
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
        __pa_schema=arrow_schema,
    )

    if selected_fields is not None:
        col_idxs = {
            arrow_schema.get_field_index(field_name) for field_name in selected_fields
        }
        exprs = make_col_ref_exprs(col_idxs, plan)
        empty_df = empty_df[list(selected_fields)]
        plan = LazyPlan(
            "LogicalProjection",
            empty_df,
            plan,
            exprs,
        )

    if limit is not None:
        plan = LazyPlan(
            "LogicalLimit",
            empty_df,
            plan,
            limit,
        )

    return wrap_plan(plan=plan)


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
    infer_datetime_format=lib.no_default,
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

    # Initialize shared metadata
    dtype = pd.ArrowDtype(pa.timestamp("ns"))
    index = arg.head(0).index
    new_metadata = pd.Series(
        dtype=dtype,
        index=index,
    )
    in_kwargs = {
        "errors": errors,
        "dayfirst": dayfirst,
        "yearfirst": yearfirst,
        "utc": utc,
        "format": format,
        "exact": exact,
        "unit": unit,
        "infer_datetime_format": infer_datetime_format,
        "origin": origin,
        "cache": cache,
    }

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
        )

    # 2. Series Case
    return _get_series_python_func_plan(
        arg._plan,
        new_metadata,
        "pandas.to_datetime",
        (),
        in_kwargs,
        is_method=False,
    )


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
