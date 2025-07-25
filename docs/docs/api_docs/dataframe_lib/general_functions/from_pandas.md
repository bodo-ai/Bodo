# bodo.pandas.from_pandas

``` py
bodo.pandas.from_pandas(df: pandas.DataFrame) -> BodoDataFrame
```

Converts a Pandas DataFrame into an equivalent BodoDataFrame.

<p class="api-header">Parameters</p>

: __df : *pandas.DataFrame*:__ The Pandas DataFrame to use as data source.

<p class="api-header">Returns</p>
: __BodoDataFrame__

<p class="api-header">Example</p>

``` py
import pandas as pd
import bodo.pandas as bodo_pd

df = pd.DataFrame(
        {
            "a": [1, 2, 3, 7] * 3,
            "b": [4, 5, 6, 8] * 3,
            "c": ["a", "b", None, "abc"] * 3,
        },
    )

bdf = bodo_pd.from_pandas(df)
print(type(bdf))
print(bdf)
```

Output:
```
<class 'bodo.pandas.frame.BodoDataFrame'>
    a  b     c
0   1  4     a
1   2  5     b
2   3  6  <NA>
3   7  8   abc
4   1  4     a
5   2  5     b
6   3  6  <NA>
7   7  8   abc
8   1  4     a
9   2  5     b
10  3  6  <NA>
11  7  8   abc
```

---