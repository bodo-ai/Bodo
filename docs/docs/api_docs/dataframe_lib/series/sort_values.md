# bodo.pandas.BodoSeries.sort\_values
``` py
BodoSeries.sort_values(
        self,
        *,
        axis: Axis = 0,
        ascending: bool = True,
        inplace: bool = False,
        kind: SortKind = "quicksort",
        na_position: str = "last",
        ignore_index: bool = False,
        key: ValueKeyFunc | None = None,
    ) -> BodoSeries
```
Sorts the elements of the BodoSeries and returns a new sorted BodoSeries.

<p class="api-header">Parameters</p>

: __ascending : *bool, default True*:__ If True, sort values in ascending order, otherwise descending.

: __na_position: *str, default 'last'*:__ Argument ‘first’ puts NaNs at the beginning, ‘last’ puts NaNs at the end.

: All other parameters will trigger a fallback to [`pandas.Series.sort_values`](https://pandas.pydata.org/docs/reference/api/pandas.Series.sort_values.html#pandas.Series.sort_values) if a non-default value is provided.

<p class="api-header">Returns</p>

: __BodoSeries__

<p class="api-header">Example</p>

``` py
import bodo.pandas as bd

bdf = bd.DataFrame(
    {
        "A": bd.array([1, 2, 3, 7] * 3, "Int64"),
        "B": ["A1", "B1", "C1", "Abc"] * 3,
        "C": bd.array([4, 5, 6, -1] * 3, "Int64"),
    }
)

sa = bdf["A"]
sa_sorted = sa.sort_values(ascending=False)
print(sa_sorted)
```

Output:
```
0     7
1     7
2     7
3     3
4     3
5     3
6     2
7     2
8     2
9     1
10    1
11    1
Name: A, dtype: int64[pyarrow]
```

---