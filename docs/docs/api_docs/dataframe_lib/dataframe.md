# DataFrame API

## bodo.pandas.BodoDataFrame.apply
``` py
BodoDataFrame.apply(
        func,
        axis=0,
        raw=False,
        result_type=None,
        args=(),
        by_row="compat",
        engine="python",
        engine_kwargs=None,
        **kwargs,
    ) -> BodoSeries
```

Apply a function along an axis of the BodoDataFrame.

Currently only supports applying a function that returns a scalar value for each row (i.e. `axis=1`).
All other uses will fall back to Pandas.
See [`pandas.DataFrame.apply`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply) for more details.

!!! note
    Calling `BodoDataFrame.apply` will immediately execute a plan to generate a small sample of the BodoDataFrame
    and then call `pandas.DataFrame.apply` on the sample to infer output types
    before proceeding with lazy evaluation.

<p class="api-header">Parameters</p>

: __func : *function*:__ Function to apply to each row.

: __axis : *{0 or 1}, default 0*:__ The axis to apply the function over. `axis=0` will fall back to [`pandas.DataFrame.apply`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply).

: __args : *tuple*:__ Additional positional arguments to pass to *func*.

: __\*\*kwargs:__ Additional keyword arguments to pass as keyword arguments to *func*.


: All other parameters will trigger a fallback to [`pandas.DataFrame.apply`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply) if a non-default value is provided.

<p class="api-header">Returns</p>
: __BodoSeries:__ The result of applying *func* to each row in the BodoDataFrame.

<p class="api-header">Example</p>

``` py
import pandas as pd
import bodo.pandas as bodo_pd

df = pd.DataFrame(
        {
            "a": pd.array([1, 2, 3] * 4, "Int64"),
            "b": pd.array([4, 5, 6] * 4, "Int64"),
            "c": ["a", "b", "c"] * 4,
        },
    )
bdf = bodo_pd.from_pandas(df)
out_bodo = bdf.apply(lambda x: x["a"] + 1, axis=1)

print(type(out_bodo))
print(out_bodo)
```

Output:
```
<class 'bodo.pandas.series.BodoSeries'>
0     2
1     3
2     4
3     2
4     3
5     4
6     2
7     3
8     4
9     2
10    3
11    4
dtype: int64[pyarrow]
```

---

## bodo.pandas.BodoDataFrame.head
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

## bodo.pandas.BodoDataFrame.map_partitions
``` py
BodoDataFrame.map_partitions(func, *args, **kwargs) -> BodoSeries | BodoDataFrame
```

Apply a function to groups of rows in a DataFrame and return a DataFrame or Series of the same size.

<p class="api-header">Parameters</p>

: __func : *Callable*:__ A function that takes in a DataFrame and returns a DataFrame or Series (with the same number of rows). Currently, functions that return a DataFrame will trigger execution, but the function will be evaluated in parallel across all workers.

: __\*args:__ Additional positional arguments to pass to *func*.

: __\*\*kwargs:__ Additional keyword arguments to pass as keyword arguments to *func*.

<p class="api-header">Returns</p>

: __BodoSeries__ or __BodoDataFrame__:  The result of applying *func* to the BodoDataFrame.

<p class="api-header">Example</p>

``` py
import bodo.pandas as bodo_pd
import pandas as pd

df = pd.DataFrame(
    {"foo": range(15), "bar": range(15, 30)}
   )

bdf = bodo_pd.from_pandas(df)

bdf.map_parititions(lambda df_: df_.foo + df_.bar)
```

Output:
```
0     15
1     17
2     19
3     21
4     23
5     25
6     27
7     29
8     31
9     33
10    35
11    37
12    39
13    41
14    43
dtype: int64[pyarrow]
```

---

## Setting DataFrame Columns

Bodo DataFrames support setting columns lazily when the value is a Series created from the same DataFrame or a constant value.
Other cases will fallback to Pandas.

<p class="api-header">Examples</p>

``` py
import bodo.pandas as bodo_pd
import pandas as pd

df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 7] * 3, "Int64"),
            "B": ["A1", "B1 ", "C1", "Abc"] * 3,
            "C": pd.array([4, 5, 6, -1] * 3, "Int64"),
        }
    )

bdf = bodo_pd.from_pandas(df)

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
import bodo.pandas as bodo_pd
import pandas as pd

df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 7] * 3, "Int64"),
            "B": ["A1", "B1 ", "C1", "Abc"] * 3,
            "C": pd.array([4, 5, 6, -1] * 3, "Int64"),
        }
    )

bdf = bodo_pd.from_pandas(df)

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

## bodo.pandas.BodoDataFrame.sort\_values
``` py
BodoDataFrame.sort_values(
        self,
        by: IndexLabel,
        *,
        axis: Axis = 0,
        ascending: bool | list[bool] | tuple[bool, ...] = True,
        inplace: bool = False,
        kind: SortKind = "quicksort",
        na_position: str | list[str] | tuple[str, ...] = "last",
        ignore_index: bool = False,
        key: ValueKeyFunc | None = None,
    ) -> BodoDataFrame
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
import bodo.pandas as bodo_pd
import pandas as pd

df = pd.DataFrame(
    {
        "A": pd.array([1, 2, 3, 7] * 3, "Int64"),
        "B": ["A1", "B1", "C1", "Abc"] * 3,
        "C": pd.array([6, 5, 4] * 4, "Int64"),
    }
)

bdf = bodo_pd.from_pandas(df)
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
