# bodo.pandas.BodoSeries.head
```
BodoSeries.head(n=5) -> BodoSeries
```

Returns the first *n* rows of the BodoSeries.

<p class="api-header">Parameters</p>

: __n : *int, default 5*:__ Number of elements to select.

<p class="api-header">Returns</p>

: __BodoSeries__

<p class="api-header">Example</p>

``` py
import bodo.pandas as bd

bdf = bd.DataFrame(
        {
            "A": bd.array([1, 2, 3, 7] * 3, "Int64"),
        }
    )

bodo_ser_head = bdf.A.head(3)
print(type(bodo_ser_head))
print(bodo_ser_head)
```

Output:

```
<class 'pandas.core.series.Series'>
0    1
1    2
2    3
Name: A, dtype: Int64
```

---