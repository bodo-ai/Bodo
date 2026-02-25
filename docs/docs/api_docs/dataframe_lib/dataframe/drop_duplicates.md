
# bodo.pandas.BodoDataFrame.drop\_duplicates
``` py
BodoDataFrame.drop_duplicates(
        subset=None,
        *,
        keep="first",
        inplace=False,
        ignore_index=False,
    ) -> BodoDataFrame
```
**GPU:** ✔ Supported

Return DataFrame with duplicate rows removed.

Currently only supports the `subset` argument and the default, `first` and `last` arguments to `keep`.
All other uses will fall back to Pandas.
See [`pandas.DataFrame.drop_duplicates`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html#pandas.DataFrame.drop_duplicates) for more details.

!!! note
    When `subset` is specified, `BodoDataFrame.drop_duplicates` is not guaranteed to produce the exact same output of Pandas.
    Instead, Bodo DataFrames will return the first or last value for the specified `subset` key that is encountered during
    processing which may occur in any order.  The set of unique keys returned will be identical to Pandas but the other values
    may be different.

<p class="api-header">Parameters</p>

: __subset : *None | List[str], default None*:__ Only consider certain columns for identifying duplicates, by default (None) use all of the columns.

: __keep : *str, default 'first'*:__ Determines which duplicates (if any) to keep.  Only 'first' and 'last' are supported.  First and last occurrences are relative to the Bodo DataFrame workers processing the dataframe and not to any index ordering of the dataframe.

: All other parameters will trigger a fallback to [`pandas.DataFrame.drop_duplicates`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html#pandas.DataFrame.drop_duplicates) if a non-default value is provided.

<p class="api-header">Returns</p>
: __BodoDataFrame:__ Bodo DataFrame with duplicates removed.

<p class="api-header">Example</p>

``` py
import bodo.pandas as pd

df = pd.DataFrame({
    'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
    'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
    'rating': [4, 4, 3.5, 15, 5]
})
print(df.drop_duplicates())
```

Output:
```
     brand style  rating
0  Indomie  pack    15.0
1  Yum Yum   cup     4.0
2  Indomie   cup     3.5
3  Indomie  pack     5.0
```

To remove duplicates on specific column(s), use `subset`.
``` py
print(df.drop_duplicates(subset=['brand']))
```

Output:
```
     brand style  rating
0  Indomie   cup     3.5
1  Yum Yum   cup     4.0
```
---
