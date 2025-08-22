# Using Regular Python inside JIT with @bodo.wrap_python {#objmode}

Regular Python functions and Bodo JIT functions can be used together in
applications arbitrarily, but there are cases where regular Python code
needs to be used *inside* JIT code. For example, you may want to use
Bodo's parallel constructs with some code that does not have JIT
support yet. The *@bodo.wrap_python* decorator allows calling regular Python functions
without jitting them. The main requirement is
that the user has to specify the data type of the function's output.

For example, the following code calls a non-JIT function on rows of a
distributed dataframe. The function decorated with `@bodo.wrap_python` runs as
regular Python and its output data type is annotated as `float64`.


``` py
import pandas as pd
import numpy as np
import bodo
import scipy.special as sc


@bodo.wrap_python("float64")
def my_non_jit_function(r):
    return np.log(r.A) + sc.entr(r.B)


@bodo.jit
def objmode_example(n):
    df = pd.DataFrame({"A": np.random.ranf(n), "B": np.arange(n)})
    df["C"] = df.apply(my_non_jit_function, axis=1)
    print(df["C"].sum())


objmode_example(10)
```


## Output Type Specification

There are various ways to specify the output data types in `wrap_python`. Basic
data types such as `float64` and `int64` can be specified as string
values (as in the previous example). For more complex data types like
dataframes, `bodo.typeof()` can be used on sample data that has the same
type as expected outputs. For example:

``` py
df_sample = pd.DataFrame({"A": [0], "B": ["AB"]}, index=[0])
df_type = bodo.typeof(df_sample)


@bodo.wrap_python(df_type)
def g():
    return pd.DataFrame({"A": [1, 2, 3], "B": ["ab", "bc", "cd"]}, index=[3, 2, 1])


@bodo.jit
def f():
    df = g()
    return df
```

This is equivalent to creating the `DataFrameType` directly:

``` py
df_type = bodo.DataFrameType(
    (bodo.types.int64[::1], bodo.string_array_type),
    bodo.NumericIndexType(bodo.types.int64),
    ("A", "B"),
)
```

The data type can be registered in Bodo so it can be referenced using a
string name later:

``` py
df_sample = pd.DataFrame({"A": [0], "B": ["AB"]}, index=[0])
bodo.register_type("my_df_type", bodo.typeof(df_sample))


@bodo.wrap_python("my_df_type")
def g():
    return pd.DataFrame({"A": [1, 2, 3], "B": ["ab", "bc", "cd"]}, index=[3, 2, 1])
```

See [pandas datatypes][pandas-dtype] for more details on
Bodo data types in general.

## What Can Be Done Inside @bodo.wrap_python

The code inside `@bodo.wrap_python` runs in regular Python,
which means `@bodo.wrap_python` does not include Bodo compiler's
automatic parallel communication management. Therefore, the computation
inside `@bodo.wrap_python` should be independent on different processors and not
require communication. In general:

-   Operations on scalars are safe
-   Operations that compute on rows independently are safe
-   Operations that compute across rows may not be safe

The example below demonstrates a valid use of `@bodo.wrap_python`, since it uses
`df.apply(axis=1)` which runs on different rows
independently. 

``` py
df_type = bodo.typeof(pd.DataFrame({"A": [1], "B": [1], "C": [1]}))

@bodo.wrap_python(df_type)
def f(df):
    return df.assign(C=df.apply(lambda r: r.A + r.B, axis=1))


@bodo.jit
def valid_objmode():
    df = pd.read_parquet("in_file.pq")
    df2 = f(df)
    df2.to_parquet("out_file.pq")


valid_objmode()
```

In contrast, the example below demonstrates an invalid use of `@bodo.wrap_python`.
The reason is that groupby computation requires grouping together
all rows with the same key across all chunks. However, on each
processor, Bodo passes a chunk of `df` to `@bodo.wrap_python` which
returns results from local groupby computation. Therefore,
`df2` does not include valid global groupby output.

``` py
df_type = bodo.typeof(pd.DataFrame({"A": [1], "B": [1]}))


@bodo.wrap_python(df_type)
def f(df):
    return df.groupby("A", as_index=False).sum()


@bodo.jit
def invalid_objmode():
    df = pd.read_parquet("in_file.pq")
    # Invalid use of wrap_python
    df2 = f(df)
    df2.to_parquet("out_file.pq")


invalid_objmode()
```

[//]: # (TODO: Uncomment when installation of prophet is resolved or a different example is produced)

[//]: # (## Groupby/Apply Object Mode Pattern)

[//]: # ()
[//]: # (ML algorithms and other complex data science computations are often)

[//]: # (called on groups of dataframe rows. Bodo supports parallelizing these)

[//]: # (computations &#40;which may not have JIT support yet&#41; using Object Mode)

[//]: # (inside `groupby/apply`. For example, the code below runs)

[//]: # ([Prophet]&#40;https://facebook.github.io/prophet/&#41; on groups of rows. This)

[//]: # (is a valid use of Object Mode since Bodo handles shuffle communication)

[//]: # (for groupby/apply and brings all rows of each group in the same local)

[//]: # (chunk. Therefore, the apply function running in Object Mode has all the)

[//]: # (data it needs.)

[//]: # ()
[//]: # (``` py)

[//]: # (import bodo)

[//]: # (import pandas as pd)

[//]: # (import numpy as np)

[//]: # ()
[//]: # (from orbit.models.dlt import DLTFull)

[//]: # ()
[//]: # (orbit_output_type = bodo.typeof&#40;pd.DataFrame&#40;{"ds": pd.date_range&#40;"2017-01-03", periods=1&#41;, "yhat": [0.0]}&#41;&#41;)

[//]: # ()
[//]: # (def run_orbit&#40;df&#41;:)

[//]: # (    m = DLTFull&#40;response_col="yhat", date_col="ds"&#41;)

[//]: # (    m.fit&#40;df&#41;)

[//]: # (    return m.predict&#40;df&#41;)

[//]: # ()
[//]: # ()
[//]: # (@bodo.jit)

[//]: # (def apply_func&#40;df&#41;:)

[//]: # (    with bodo.objmode&#40;df2=orbit_output_type&#41;:)

[//]: # (        df2 = run_orbit&#40;df&#41;)

[//]: # (    return df2)

[//]: # ()
[//]: # ()
[//]: # (@bodo.jit)

[//]: # (def f&#40;df&#41;:)

[//]: # (    df2 = df.groupby&#40;"A"&#41;.apply&#40;apply_func&#41;)

[//]: # (    return df2)

[//]: # ()
[//]: # ()
[//]: # (n = 10)

[//]: # (df = pd.DataFrame&#40;{"A": np.arange&#40;n&#41; % 3, "ds": pd.date_range&#40;"2017-01-03", periods=n&#41;, "y": np.arange&#40;n&#41;}&#41;)

[//]: # (print&#40;f&#40;df&#41;&#41;)

[//]: # (```)

[//]: # ()
