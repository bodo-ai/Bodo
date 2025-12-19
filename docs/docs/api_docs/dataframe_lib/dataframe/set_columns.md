# Setting DataFrame Columns

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
