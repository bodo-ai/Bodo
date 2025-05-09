# Bodo Pandas API (Bodo DataFrame Library)

The Bodo DataFrame Library is designed to accelerate and scale Pandas workflows with just a one-line change — simply replace:

``` py
import pandas as pd
```

with

``` py
import bodo.pandas as pd
```

and your existing code can immediately take advantage of high-performance, scalable execution.

Key features include:

- __Full Pandas compatibility__ with a transparent fallback mechanism to native Pandas,
ensuring that your workflows continue uninterrupted even if a feature is not yet supported.

- __Advanced query optimization__ such as
 filter pushdown, column pruning and join reordering behind the scenes.

- __Scalable MPI-based execution__, leveraging High-Performance Computing (HPC) techniques for efficient parallelism;
whether you're working on a laptop or running jobs across a large cloud cluster.

- __Vectorized execution__ with streaming and spill-to-disk capabilities,
making it possible to process datasets larger than memory reliably.

!!! warning
    Bodo DataFrame Library is under active development and is currently considered experimental.
    Some features and APIs may not yet be fully supported.
    We welcome your feedback — please join our community [Slack](https://bodocommunity.slack.com/join/shared_invite/zt-qwdc8fad-6rZ8a1RmkkJ6eOX1X__knA#/shared-invite/email) or open an issue on [our GitHub](https://github.com/bodo-ai/Bodo)
    if you encounter any problems!

## Lazy Evaluation and Fallback to Pandas

Bodo DataFrame Library operates with lazy evaluation to allow query optimization, meaning operations are recorded into a query plan rather than executed immediately.
Execution is automatically triggered only when results are actually needed, such as when displaying the first 5 rows in a DataFrame with `print(df.head(5))`.

If the user code encounters an unsupported Pandas API or an unsupported parameter, Bodo DataFrame library gracefully falls back to native Pandas.
When this happens, the query plan is immediately executed, the resulting data is collected onto a single core and converted to a Pandas DataFrame, and further operations proceed using Pandas.

!!! warning
    Fallback to Pandas may lead to degraded performance and increase the risk of out-of-memory (OOM) errors, especially for large datasets.


## General Functions

### bodo.pandas.from_pandas

``` py
bodo.pandas.from_pandas(df: pandas.DataFrame) -> BodoDataFrame
```

Converts a Pandas DataFrame into an equivalent BodoDataFrame.

<p style="font-size: 1.1em; font-weight: bold;">Parameters</p>

: __df : *pandas.DataFrame*:__ The Pandas DataFrame to use as data source.

<p style="font-size: 1.1em; font-weight: bold;">Returns</p>
: __BodoDataFrame__

<p style="font-size: 1.1em; font-weight: bold;">Example</p>

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

## Input/Output

### bodo.pandas.read_parquet
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

<p style="font-size: 1.1em; font-weight: bold;">Parameters</p>

: __path : *str, list[str]*:__ Location of the parquet file(s) to read.
Refer to [`pandas.read_parquet`](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html#pandas.read_parquet) for more details.
The type of this argument differs from Pandas.

: All other parameters will trigger a fallback to [`pandas.read_parquet`](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html#pandas.read_parquet) if a non-default value is provided.

<p style="font-size: 1.1em; font-weight: bold;">Returns</p>
: __BodoDataFrame__

<p style="font-size: 1.1em; font-weight: bold;">Example</p>

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


## DataFrame API

### bodo.pandas.BodoDataFrame.apply
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
    Calling `BodoDataFrame.apply` will immediately execute a plan to generate the a small sample of the BodoDataFrame
    and then call `pandas.DataFrame.apply` on the head to infer output types
    before proceeding with lazy evaluation.

<p style="font-size: 1.1em; font-weight: bold;">Parameters</p>

: __func : *function*:__ Function to apply to each row.

: __axis : *{0 or 1}, default 0*:__ The axis to apply the function over. `axis=0` will fall back to [`pandas.DataFrame.apply`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply).

: All other parameters will trigger a fallback to [`pandas.DataFrame.apply`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply) if a non-default value is provided.

<p style="font-size: 1.1em; font-weight: bold;">Returns</p>
: __BodoSeries:__ The result of applying *func* to each row in the BodoDataFrame.

<p style="font-size: 1.1em; font-weight: bold;">Example</p>

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

### bodo.pandas.BodoDataFrame.head
``` py
BodoDataFrame.head(n=5) -> BodoDataFrame
```

Returns the first *n* rows of the BodoDataFrame.

<p style="font-size: 1.1em; font-weight: bold;">Parameters</p>

: __n : *int, default 5*:__ Number of rows to select.

<p style="font-size: 1.1em; font-weight: bold;">Returns</p>

: __BodoDataFrame__

<p style="font-size: 1.1em; font-weight: bold;">Example</p>

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
print(restored_df.head(2))
```

Output:
```
<class 'bodo.pandas.frame.BodoDataFrame'>
   foo  bar
0    0   15
1    1   16
```

---

### Setting BodoDataFrame Columns

BodoDataFrames support setting columns lazily when the value is a projection from the same DataFrame.
Other cases will fallback to Pandas.

<p style="font-size: 1.1em; font-weight: bold;">Example</p>

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

---

## Series API

### bodo.pandas.BodoSeries.head
```
BodoSeries.head(n=5) -> BodoSeries
```

Returns the first *n* rows of the BodoSeries.

<p style="font-size: 1.1em; font-weight: bold;">Parameters</p>

: __n : *int, default 5*:__ Number of elements to select.

<p style="font-size: 1.1em; font-weight: bold;">Returns</p>

: __BodoSeries__

<p style="font-size: 1.1em; font-weight: bold;">Example</p>

``` py
import bodo.pandas as bodo_pd
import pandas as pd

df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 7] * 3, "Int64"),
        }
    )

bdf = bodo_pd.from_pandas(df)
bodo_ser_head = df.A.head(3)
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

### bodo.pandas.BodoSeries.map
```
BodoSeries.map(arg, na_action=None) -> BodoSeries
```
Map values of a BodoSeries according to a mapping.

!!! note
    Calling `BodoSeries.map` will immediately execute a plan to generate a small sample of the BodoSeries
    and then call `pandas.Series.map` on the head to infer output types
    before proceeding with lazy evaluation.

<p style="font-size: 1.1em; font-weight: bold;">Parameters</p>

: __arg : *function, collections.abc.Mapping subclass or Series*:__ Mapping correspondence.

: __na_actions:__ will fall back to [`pandas.Series.map`](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html#pandas.Series.map) if 'ignore' is provided.

<p style="font-size: 1.1em; font-weight: bold;">Returns</p>

: __BodoSeries__

<p style="font-size: 1.1em; font-weight: bold;">Example</p>

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
bodo_ser = bdf.A.map(lambda x: x ** 2)
print(type(bodo_ser))
print(bodo_ser)
```

Output:
```
<class 'bodo.pandas.series.BodoSeries'>
0      1
1      4
2      9
3     49
4      1
5      4
6      9
7     49
8      1
9      4
10     9
11    49
Name: A, dtype: int64[pyarrow]
```

---

### bodo.pandas.BodoSeries.str.lower
```
BodoSeries.str.lower() -> BodoSeries
```
Converts strings in a BodoSeries to lowercase.
Equivalent to [`str.lower()`](https://docs.python.org/3/library/stdtypes.html#str.lower).

<p style="font-size: 1.1em; font-weight: bold;">Returns</p>

: __BodoSeries__

<p style="font-size: 1.1em; font-weight: bold;">Example</p>

``` py
import bodo.pandas as bodo_pd
import pandas as pd

df = pd.DataFrame(
        {
            "A": ["A1", "B1", "C1", "Abc"] * 3,
        }
    )

bdf = bodo_pd.from_pandas(df)
bodo_ser = bdf.A.str.lower()
print(type(bodo_ser))
print(bodo_ser)
```

Output:

```
<class 'bodo.pandas.series.BodoSeries'>
0      a1
1      b1
2      c1
3     abc
4      a1
5      b1
6      c1
7     abc
8      a1
9      b1
10     c1
11    abc
Name: A, dtype: string
```

---

### bodo.pandas.BodoSeries.str.strip
```
BodoSeries.str.strip(to_strip=None) -> BodoSeries
```
Remove leading and trailing characters.
Equivalent to [`str.strip()`](https://docs.python.org/3/library/stdtypes.html#str.strip).

<p style="font-size: 1.1em; font-weight: bold;">Parameters</p>

: __to_strip:__
Will fall back to [`pandas.Series.str.strip`](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.strip.html#pandas.Series.str.strip) if a value other than None is provided.

<p style="font-size: 1.1em; font-weight: bold;">Returns</p>

: __BodoSeries__

<p style="font-size: 1.1em; font-weight: bold;">Example</p>

``` py
import bodo.pandas as bodo_pd
import pandas as pd

df = pd.DataFrame(
        {
            "A": [" \t A1\n", "\n\nB1 \t", "C1\n", "\t\nAbc"] * 3,
        }
    )

bdf = bodo_pd.from_pandas(df)
bodo_ser = bdf.A.str.strip()
print(type(bodo_ser))
print(bodo_ser)
```

Output:

```
<class 'bodo.pandas.series.BodoSeries'>
0      A1
1      B1
2      C1
3     Abc
4      A1
5      B1
6      C1
7     Abc
8      A1
9      B1
10     C1
11    Abc
Name: A, dtype: string
```
