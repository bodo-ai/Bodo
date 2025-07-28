# bodo.pandas.BodoDataFrame.to\_iceberg
``` py
BodoDataFrame.to_iceberg(
        table_identifier,
        catalog_name=None,
        *,
        catalog_properties=None,
        location=None,
        append=False,
        partition_spec=None,
        sort_order=None,
        properties=None,
        snapshot_properties=None
)
```
Write a DataFrame as an Iceberg dataset.

Refer to [`pandas.DataFrame.to_iceberg`](https://pandas.pydata.org/docs/dev/reference/api/pandas.DataFrame.to_iceberg.html) for more details.

!!! warning
    This function is experimental in Pandas and may change in future releases.


!!! note
    This function assumes that the Iceberg namespace is already created in the catalog.

<p class="api-header">Parameters</p>

: __table_identifier: *str*:__ Table identifier to write
: __catalog_name: *str, optional*:__ Name of the catalog to use. If not provided, the default catalog will be used. See [PyIceberg's documentation](https://py.iceberg.apache.org/#connecting-to-a-catalog) for more details.
: __catalog_properties: *dict[str, Any], optional*:__ Properties for the catalog connection.
: __location: *str, optional*:__ Location of the table (if supported by the catalog). If this is passed a path and catalog_name and catalog_properties are None, it will use a filesystem catalog with the provided location. If the location is an S3 Tables ARN it will use the S3TablesCatalog.
: __append: *bool*:__ Append or overwrite if the table exists
: __partition_spec: *PartitionSpec, optional*:__ PyIceberg partition spec for the table (only used if creating a new table). See [PyIceberg's documentation](https://py.iceberg.apache.org/api/#partitions) for more details.
: __sort_order: *SortOrder, optional*:__ PyIceberg sort order for the table (only used if creating a new table). See [PyIceberg's documentation](https://py.iceberg.apache.org/reference/pyiceberg/table/sorting/#pyiceberg.table.sorting.SortOrder) for more details.
: __properties: *dict[str, Any], optional*:__ Properties to add to the new table.
: __snapshot_properties: *dict[str, Any], optional*:__ Properties to add to the new table snapshot.

<p class="api-header">Example</p>


Simple write of a table on the filesystem without a catalog:
``` py
import bodo.pandas as bd
from pyiceberg.transforms import IdentityTransform
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.table.sorting import SortField, SortOrder

bdf = bd.DataFrame(
        {
            "one": [-1.0, 1.3, 2.5, 3.0, 4.0, 6.0, 10.0],
            "two": ["foo", "bar", "baz", "foo", "bar", "baz", "foo"],
            "three": [True, False, True, True, True, False, False],
            "four": [-1.0, 5.1, 2.5, 3.0, 4.0, 6.0, 11.0],
            "five": ["foo", "bar", "baz", None, "bar", "baz", "foo"],
        }
    )

part_spec = PartitionSpec(PartitionField(2, 1001, IdentityTransform(), "id_part"))
sort_order = SortOrder(SortField(source_id=4, transform=IdentityTransform()))
bdf.to_iceberg("test_table", location="./iceberg_warehouse", partition_spec=part_spec, sort_order=sort_order)

out_df = bd.read_iceberg("test_table", location="./iceberg_warehouse")
# Only reads Parquet files of partition "foo" from storage
print(out_df[out_df["two"] == "foo"])
```

Output:
```
    one  two  three  four  five
0  -1.0  foo   True  -1.0   foo
1   3.0  foo   True   3.0  <NA>
2  10.0  foo  False  11.0   foo
```

Write a DataFrame to an Iceberg table in S3 Tables using the location parameter:

``` py
df.to_iceberg(
    table_identifier="my_table",
    location="arn:aws:s3tables:<region>:<account_number>:my-bucket/my-table"
)
```

---
