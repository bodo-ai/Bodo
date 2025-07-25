# bodo.pandas.BodoDataFrame.sort\_values
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