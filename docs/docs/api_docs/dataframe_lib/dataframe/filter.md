# bodo.pandas.BodoDataFrame.filter
``` py
BodoDataFrame.filter(items=None, like=None, regex=None, axis=None)
```
Subset the BodoDataFrame according to the specified (column) labels. This does not raise an error if a provided label is not in the dataframe.

<p class="api-header">Parameters</p>

: __items: *list-like, default None*:__ Keep column labels that are in `items`.

: __like : *str, default None*:__ Keep column labels for which `like` is found in the column label string.

: __regex: *str (regular expression), default None*:__ Keep column labels for which re.search(`regex`, label) == True.

: Exactly one of the above parameters must be provided, else a `TypeError` will be raised.

: Setting the `axis` parameter will trigger a fallback to [`pandas.DataFrame.filter`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.filter.html) if a value other than `None` (representing the column axis) is provided.

<p class="api-header">Returns</p>

: __BodoDataFrame__

<p class="api-header">Example</p>

Using `items`:

``` py
import bodo.pandas as bd

bdf = bd.DataFrame(
    {
        "A": [0, 1, 2, 3, 4],
        "BB": [5, 6, 7, 8, 9],
        "CAC": [10, 11, 12, 13, 14]
    }
)

bdf_filtered = bdf.filter(items=("BB", "CAC", "D"))
print(bdf_filtered)
```

Output:
```
   BB  CAC
0   5   10
1   6   11
2   7   12
3   8   13
4   9   14
```

Using `like`:

``` py
import bodo.pandas as bd

bdf = bd.DataFrame(
    {
        "A": [0, 1, 2, 3, 4],
        "BB": [5, 6, 7, 8, 9],
        "CAC": [10, 11, 12, 13, 14]
    }
)

bdf_filtered = bdf.filter(like="A")
print(bdf_filtered)
```

Output:
```
   A  CAC
0  0   10
1  1   11
2  2   12
3  3   13
4  4   14
```

---