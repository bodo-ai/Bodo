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
**GPU:** ✔ Supported

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
import bodo.pandas as bd

original_df = bd.DataFrame(
    {"foo": range(15), "bar": range(15, 30)}
   )

original_df.to_parquet("example.pq")

restored_df = bd.read_parquet("example.pq")
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

!!! tip "GPU Acceleration"
    May fall back to CPU if the plan includes operations such as `head()` that require Pandas sampling.

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
    location: str | None = None,
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
: __location: *str, optional*:__ Location of the table (if supported by the catalog). If this is passed a path and catalog_name and catalog_properties are None, it will use a filesystem catalog with the provided location. If the location is an S3 Tables ARN it will use the S3TablesCatalog.

: Non-default values for case_sensitive and scan_properties will trigger a fallback to [`pandas.read_iceberg`](https://pandas.pydata.org/docs/dev/reference/api/pandas.read_iceberg.html).


<p class="api-header">Returns</p>
: __BodoDataFrame__

<p class="api-header">Examples</p>

Simple read of a table stored without a catalog on the filesystem:
``` py
import bodo.pandas as bd

df = bd.read_iceberg("my_table", location="s3://path/to/iceberg/warehouse")
```


Read a table using a predefined PyIceberg catalog.
``` py
import bodo.pandas as bd

df = bd.read_iceberg(
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
import bodo.pandas as bd
import pyiceberg.catalog

df = bd.read_iceberg(
    table_identifier="my_schema.my_table",
    catalog_properties={
        pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
        pyiceberg.catalog.WAREHOUSE_LOCATION: path_to_warehouse_dir,
    }
)
```

Read a table from an S3 Tables Bucket using the location parameter.

``` py
import bodo.pandas as bd

df = bd.read_iceberg(
    table_identifier="my_table",
    location="arn:aws:s3tables:<region>:<account_number>:my-bucket/my-table"
)
```


## bodo.pandas.read_iceberg_table
``` py
bodo.pandas.read_iceberg_table(
    table: pyiceberg.table.Table,
) -> BodoDataFrame
```

Creates a BodoDataFrame object for reading from an Iceberg table lazily.

!!! warning
    This function is not part of the Pandas API and is specific to Bodo.

<p class="api-header">Parameters</p>

: __table_identifier: *pyiceberg.table.Table*:__ PyIceberg Table object to read with Bodo.

<p class="api-header">Returns</p>
: __BodoDataFrame__

<p class="api-header">Examples</p>

Simple read of a local table stored in a sql catalog:
``` py
from pyiceberg.catalog import load_catalog

warehouse_path = "/tmp/warehouse"
catalog = load_catalog(
    "default",
    **{
        'type': 'sql',
        "uri": f"sqlite:///{warehouse_path}/pyiceberg_catalog.db",
        "warehouse": f"file://{warehouse_path}",
    },
)
table = catalog.load_table("my_schema.my_table")
df = bodo.pandas.read_iceberg_table(table)
```
