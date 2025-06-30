# DataFrame API

## BodoDataFrame.apply
``` py
BodoDataFrame.apply(
        func,
        axis=0,
        raw=False,
        result_type=None,
        args=(),
        by_row="compat",
        engine="python",
        engine_kwargs=None,
        **kwargs,
    ) -> BodoSeries
```

Apply a function along an axis of the BodoDataFrame.

Currently only supports applying a function that returns a scalar value for each row (i.e. `axis=1`).
All other uses will fall back to Pandas.
See [`pandas.DataFrame.apply`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply) for more details.

!!! note
    Calling `BodoDataFrame.apply` will immediately execute a plan to generate a small sample of the BodoDataFrame
    and then call `pandas.DataFrame.apply` on the sample to infer output types
    before proceeding with lazy evaluation.

<p class="api-header">Parameters</p>

: __func : *function*:__ Function to apply to each row.

: __axis : *{0 or 1}, default 0*:__ The axis to apply the function over. `axis=0` will fall back to [`pandas.DataFrame.apply`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply).

: __args : *tuple*:__ Additional positional arguments to pass to *func*.

: __\*\*kwargs:__ Additional keyword arguments to pass as keyword arguments to *func*.


: All other parameters will trigger a fallback to [`pandas.DataFrame.apply`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply) if a non-default value is provided.

<p class="api-header">Returns</p>
: __BodoSeries:__ The result of applying *func* to each row in the BodoDataFrame.

<p class="api-header">Example</p>

``` py
import bodo.pandas as bd

bdf = bd.DataFrame(
        {
            "a": bd.array([1, 2, 3] * 4, "Int64"),
            "b": bd.array([4, 5, 6] * 4, "Int64"),
            "c": ["a", "b", "c"] * 4,
        },
    )

out_bodo = bdf.apply(lambda x: x["a"] + 1, axis=1)

print(type(out_bodo))
print(out_bodo)
```

Output:
```
<class 'bodo.pandas.series.BodoSeries'>
0     2
1     3
2     4
3     2
4     3
5     4
6     2
7     3
8     4
9     2
10    3
11    4
dtype: int64[pyarrow]
```

---

## BodoDataFrame.groupby {#frame-groupby}

``` py
BodoDataFrame.groupby(
    by=None,
    axis=lib.no_default,
    level=None,
    as_index=True,
    sort=False,
    group_keys=True,
    observed=lib.no_default,
    dropna=True
) -> DataFrameGroupBy
```

Creates a DataFrameGroupBy object representing the data in the input DataFrame grouped by a column or list of columns. The object can then be used to apply functions over groups.

<p class="api-header">Parameters</p>

: __by : *str | List[str]*:__ The column or list of columns to use when creating groups.

: __as\_index : *bool, default True*:__ Whether the grouped labels will appears as an index in the final output. If *as_index* is False, then the grouped labels will appear as regular columns.

: __dropna: *bool, default True*__ If True, rows where the group label contains a missing value will be dropped from the final output.

: All other parameters will trigger a fallback to [`pandas.DataFrame.groupby`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) if a non-default value is provided.

<p class="api-header">Returns</p>

: __DataFrameGroupBy__

<p class="api-header">Examples</p>

``` py
import bodo.pandas as bd

bdf1 = bd.DataFrame({
    "A": ["foo", "foo", "bar", "bar"],
    "B": [1, 1, 1, None],
    "C": [1, 2, 3, 4]
})

bdf2 = bdf1.groupby(["A", "B"]).sum()
print(bdf2)
```
Output:
``` py
         C
A   B
bar 1.0  3
foo 1.0  3
```
---
``` py
bdf3 = bdf1.groupby(["A", "B"], as_index=False, dropna=False).sum()
print(bdf2)
```
Output:
```
     A     B  C
0  bar  <NA>  4
1  foo   1.0  3
2  bar   1.0  3
```
---

## BodoDataFrame.head
``` py
BodoDataFrame.head(n=5) -> BodoDataFrame
```

Returns the first *n* rows of the BodoDataFrame.

<p class="api-header">Parameters</p>

: __n : *int, default 5*:__ Number of rows to select.

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
restored_df_head = restored_df.head(2)
print(type(restored_df_head))
print(restored_df_head)
```

Output:
```
<class 'bodo.pandas.frame.BodoDataFrame'>
   foo  bar
0    0   15
1    1   16
```

---

## BodoDataFrame.map_partitions
``` py
BodoDataFrame.map_partitions(func, *args, **kwargs) -> BodoSeries | BodoDataFrame
```

Apply a function to groups of rows in a DataFrame and return a DataFrame or Series of the same size.

If the input DataFrame is lazy (i.e. its plan has not been evaluated yet) and *func* returns a Series, then
the output will be lazy as well. When the lazy output is evaluated, *func* will take batches of
rows from the input DataFrame. In the cases where *func* returns a DataFrame or the input DataFrame is not lazy,
each worker will call *func* on their entire local chunk of the input DataFrame.

<p class="api-header">Parameters</p>

: __func : *Callable*:__ A function that takes in a DataFrame and returns a DataFrame or Series (with the same number of rows). Currently, functions that return a DataFrame will trigger execution even if the input DataFrame has a lazy plan.

: __\*args:__ Additional positional arguments to pass to *func*.

: __\*\*kwargs:__ Additional keyword arguments to pass as keyword arguments to *func*.

<p class="api-header">Returns</p>

: __BodoSeries__ or __BodoDataFrame__:  The result of applying *func* to the BodoDataFrame.

<p class="api-header">Example</p>

``` py
import bodo.pandas as bd

bdf = bd.DataFrame(
    {"foo": range(15), "bar": range(15, 30)}
   )

bdf_mapped = bdf.map_partitions(lambda df_: df_.foo + df_.bar)
print(bdf_mapped)
```

Output:
```
0     15
1     17
2     19
3     21
4     23
5     25
6     27
7     29
8     31
9     33
10    35
11    37
12    39
13    41
14    43
dtype: int64[pyarrow]
```

---

## Setting DataFrame Columns

Bodo DataFrames support setting columns lazily when the value is a Series created from the same DataFrame or a constant value.
Other cases will fallback to Pandas.

<p class="api-header">Examples</p>

``` py
import bodo.pandas as bd

bdf = bd.DataFrame(
        {
            "A": bd.array([1, 2, 3, 7] * 3, "Int64"),
            "B": ["A1", "B1 ", "C1", "Abc"] * 3,
            "C": bd.array([4, 5, 6, -1] * 3, "Int64"),
        }
    )

bdf["D"] = bdf["B"].str.lower()
print(type(bdf))
print(bdf.D)
```

Output:
```
<class 'bodo.pandas.frame.BodoDataFrame'>
0      a1
1     b1
2      c1
3     abc
4      a1
5     b1
6      c1
7     abc
8      a1
9     b1
10     c1
11    abc
Name: D, dtype: string
```


``` py
import bodo.pandas as bd

bdf = bd.DataFrame(
        {
            "A": bd.array([1, 2, 3, 7] * 3, "Int64"),
            "B": ["A1", "B1 ", "C1", "Abc"] * 3,
            "C": bd.array([4, 5, 6, -1] * 3, "Int64"),
        }
    )

bdf["D"] = 11
print(type(bdf))
print(bdf.D)
```

Output:
```
<class 'bodo.pandas.frame.BodoDataFrame'>
0     11
1     11
2     11
3     11
4     11
5     11
6     11
7     11
8     11
9     11
10    11
11    11
Name: D, dtype: int64[pyarrow]
```

---

## BodoDataFrame.sort\_values
``` py
BodoDataFrame.sort_values(by, *, axis=0, ascending=True, inplace=False, kind="quicksort", na_position="last", ignore_index=False, key=None)
```
Sorts the elements of the BodoDataFrame and returns a new sorted BodoDataFrame.

<p class="api-header">Parameters</p>

: __by: *str or list of str*:__ Name or list of column names to sort by.

: __ascending : *bool or list of bool, default True*:__ Sort ascending vs. descending. Specify list for multiple sort orders. If this is a list of bools, must match the length of the by.

: __na_position: *str {'first', 'last'} or list of str, default 'last'*:__ Puts NaNs at the beginning if first; last puts NaNs at the end. Specify list for multiple NaN orders by key.  If this is a list of strings, must match the length of the by.

: All other parameters will trigger a fallback to [`pandas.DataFrame.sort_values`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html) if a non-default value is provided.

<p class="api-header">Returns</p>

: __BodoDataFrame__

<p class="api-header">Example</p>

``` py
import bodo.pandas as bd

bdf = bd.DataFrame(
    {
        "A": bd.array([1, 2, 3, 7] * 3, "Int64"),
        "B": ["A1", "B1", "C1", "Abc"] * 3,
        "C": bd.array([6, 5, 4] * 4, "Int64"),
    }
)

bdf_sorted = bdf.sort_values(by=["A", "C"], ascending=[False, True])
print(bdf_sorted)
```

Output:
```
    A    B  C
0   7  Abc  4
1   7  Abc  5
2   7  Abc  6
3   3   C1  4
4   3   C1  5
5   3   C1  6
6   2   B1  4
7   2   B1  5
8   2   B1  6
9   1   A1  4
10  1   A1  5
11  1   A1  6
```

---

## BodoDataFrame.to\_parquet
``` py
BodoDataFrame.to_parquet(path=None, engine="auto", compression="snappy", index=None, partition_cols=None, storage_options=None, row_group_size=-1, **kwargs)
```
Write a DataFrame as a Parquet dataset.

<p class="api-header">Parameters</p>

: __path: *str*:__ Output path to write. It can be a local path (e.g. `output.parquet`), AWS S3 (`s3://...`), Azure ALDS (`abfs://...`, `abfss://...`), or GCP GCS (`gcs://...`, `gs://`).

: __compression : *str, default 'snappy'*:__ File compression to use. Can be None, 'snappy', 'gzip', or 'brotli'.

: __row_group_size : *int*:__ Row group size in output Parquet files. -1 allows the backend to choose.

: All other parameters will trigger a fallback to [`pandas.DataFrame.to_parquet`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html).


<p class="api-header">Example</p>

``` py
import bodo.pandas as bd

bdf = bd.DataFrame(
    {
        "A": bd.array([1, 2, 3, 7] * 3, "Int64"),
        "B": ["A1", "B1", "C1", "Abc"] * 3,
        "C": bd.array([6, 5, 4] * 4, "Int64"),
    }
)

bdf.to_parquet("output.parquet")
print(bd.read_parquet("output.parquet"))
```

Output:
```
    A    B  C
0   1   A1  6
1   2   B1  5
2   3   C1  4
3   7  Abc  6
4   1   A1  5
5   2   B1  4
6   3   C1  6
7   7  Abc  5
8   1   A1  4
9   2   B1  6
10  3   C1  5
11  7  Abc  4
```

---

## BodoDataFrame.to\_iceberg
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
