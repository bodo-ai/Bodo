
# bodo.pandas.BodoDataFrame.head
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