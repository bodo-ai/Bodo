# Input/Output

## bodo.pandas.read_parquet
``` py
bodo.pandas.read_parquet(
    path,
    engine="auto",
    columns=None,
    storage_options=None,
    use_nullable_dtypes=lib.no_default,
    dtype_backend=lib.no_default,
    filesystem=None,
    filters=None,
    **kwargs,
) -> BodoDataFrame
```

Creates a BodoDataFrame object for reading from parquet file(s) lazily.

<p class="api-header">Parameters</p>

: __path : *str, list[str]*:__ Location of the parquet file(s) to read.
Refer to [`pandas.read_parquet`](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html#pandas.read_parquet) for more details.
The type of this argument differs from Pandas.

: All other parameters will trigger a fallback to [`pandas.read_parquet`](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html#pandas.read_parquet) if a non-default value is provided.

<p class="api-header">Returns</p>
: __BodoDataFrame__

<p class="api-header">Example</p>

``` py
import bodo
import bodo.pandas as bodo_pd
import pandas as pd

original_df = pd.DataFrame(
    {"foo": range(15), "bar": range(15, 30)}
   )

@bodo.jit
def write_parquet(df):
    df.to_parquet("example.pq")

write_parquet(original_df)

restored_df = bodo_pd.read_parquet("example.pq")
print(type(restored_df))
print(restored_df.head())
```

Output:

```
<class 'bodo.pandas.frame.BodoDataFrame'>
   foo  bar
0    0   15
1    1   16
2    2   17
3    3   18
4    4   19
```

---

## bodo.pandas.read_iceberg
``` py
bodo.pandas.read_iceberg(
    table_identifier: str,
    catalog_name: str | None = None,
    catalog_properties: dict[str, Any] | None = None,
    row_filter: str | None = None,
    selected_fields: tuple[str] | None = None,
    case_sensitive: bool = True,
    snapshot_id: int | None = None,
    limit: int | None = None,
    scan_properties: dict[str, Any] | None = None,
) -> BodoDataFrame
```

Creates a BodoDataFrame object for reading from an Iceberg table lazily.

Refer to [`pandas.read_iceberg`](https://pandas.pydata.org/docs/dev/reference/api/pandas.read_iceberg.html) for more details.

!!! warning
    This function is experimental in Pandas and may change in future releases.

<p class="api-header">Parameters</p>

: __table_identifier: *str*:__ Identifier of the Iceberg table to read. This should be in the format `schema.table`
: __catalog_name: *str, optional*:__ Name of the catalog to use. If not provided, the default catalog will be used. See [PyIceberg's documentation](https://py.iceberg.apache.org/#connecting-to-a-catalog) for more details.
: __catalog_properties: *dict[str, Any], optional*:__ Properties for the catalog connection.
: __row_filter: *str, optional*:__ expression to filter rows.
: __selected_fields: *tuple[str], optional*:__ Fields to select from the table, if not provided, all fields will be selected.
: __snapshot_id: *int, optional*:__ ID of the snapshot to read from. If not provided, the latest snapshot will be used.
: __limit: *int, optional*:__ Maximum number of rows to read. If not provided, all rows will be read.

: Non-default values for case_sensitive and scan_properties will trigger a fallback to [`pandas.read_iceberg`](https://pandas.pydata.org/docs/dev/reference/api/pandas.read_iceberg.html).


<p class="api-header">Returns</p>
: __BodoDataFrame__

<p class="api-header">Example</p>

Read a table using a predefined PyIceberg catalog.
``` py
import bodo
import bodo.pandas as bodo_pd
df = bodo_pd.read_iceberg(
    table_identifier="my_schema.my_table",
    catalog_name="my_catalog",
    row_filter="col1 > 10",
    selected_fields=("col1", "col2"),
    snapshot_id=123456789,
    limit=1000
)
```

Read a table using a new PyIceberg catalog with custom properties.
``` py
import bodo
import bodo.pandas as bodo_pd
import pyiceberg.catalog
df = bodo_pd.read_iceberg(
    table_identifier="my_schema.my_table",
    catalog_properties={
        pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
        pyiceberg.catalog.WAREHOUSE_LOCATION: path_to_warehouse_dir,
    }
)
```
