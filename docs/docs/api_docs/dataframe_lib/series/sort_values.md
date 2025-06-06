# bodo.pandas.BodoSeries.sort\_values
```
BodoSeries.sort_values(ascending=True, na_position='last') -> BodoSeries
```
Sorts the elements of the BodoSeries and returns a new sorted BodoSeries.

<p class="api-header">Parameters</p>

: __ascending : *bool, default True*:__ If True, sort values in ascending order, otherwise descending.

: __na_position: *str, default 'last'*:__ Argument ‘first’ puts NaNs at the beginning, ‘last’ puts NaNs at the end.

<p class="api-header">Returns</p>

: __BodoSeries__

<p class="api-header">Example</p>

``` py
import bodo.pandas as bodo_pd
import pandas as pd

df = pd.DataFrame(
    {
        "A": pd.array([1, 2, 3, 7] * 3, "Int64"),
        "B": ["A1", "B1", "C1", "Abc"] * 3,
        "C": pd.array([4, 5, 6, -1] * 3, "Int64"),
    }
)

bdf = bodo_pd.from_pandas(df)
sa = bdf["A"]
print(sa)
sa_sorted = sa.sort_values(ascending=False)
print(sa_sorted)
```

Output:
```
0     1
1     2
2     3
3     7
4     1
5     2
6     3
7     7
8     1
9     2
10    3
11    7
Name: A, dtype: int64[pyarrow]
3     7
7     7
11    7
2     3
6     3
10    3
1     2
5     2
9     2
0     1
4     1
8     1
Name: A, dtype: int64[pyarrow]
```

---
