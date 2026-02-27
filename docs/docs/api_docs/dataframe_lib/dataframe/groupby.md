# bodo.pandas.BodoDataFrame.groupby {#frame-groupby}

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
**GPU:** ✔ Supported

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