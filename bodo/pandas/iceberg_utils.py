"""
Provides utilities for creating Bodo Lazy Plans from Iceberg tables.
"""

from __future__ import annotations

import typing as pt
from dataclasses import dataclass

import pandas as pd
import pyarrow as pa

from bodo.pandas.plan import (
    LogicalGetIcebergRead,
)
from bodo.pandas.utils import (
    arrow_to_empty_df,
)


@dataclass
class JoinFilterInfo:
    filter_ids: list[int]
    equality_filter_columns: list[list[int]]
    orig_build_key_cols: list[list[int]]
    equality_is_first_locations: list[list[bool]]


def build_iceberg_read_plan(
    table_identifier: str,
    catalog_name: str | None = None,
    catalog_properties: dict[str, pt.Any] | None = None,
    row_filter: str | None = None,
    snapshot_id: int | None = None,
    location: str | None = None,
    join_filter_info: JoinFilterInfo | None = None,
    selected_fields: list[str] | None = None,
    limit: int | None = None,
) -> tuple[LogicalGetIcebergRead, pd.DataFrame, pa.Schema]:
    """Create an Iceberg read plan for the given table and return the plan, an empty
    dataframe with the correct schema and the arrow schema
    """
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

    # None here implies all fields should be selected
    selected_idxs = None
    if selected_fields is not None:
        selected_idxs = [
            arrow_schema.get_field_index(field_name) for field_name in selected_fields
        ]
        empty_df = empty_df[list(selected_fields)]

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
        join_filter_info=join_filter_info,
        selected_fields=selected_idxs,
        limit=limit,
    )

    return plan, empty_df, arrow_schema
