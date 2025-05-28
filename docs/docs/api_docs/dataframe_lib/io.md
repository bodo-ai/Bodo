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
