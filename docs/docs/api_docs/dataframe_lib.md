# Bodo Pandas API (bodo.pandas)

The bodo.pandas library is designed to scale Pandas workflows with just a one-line change — simply replace:

``` py
import pandas as pd
```

with

``` py
import bodo.pandas as pd
```

and your existing code can immediately take advantage of high-performance, scalable execution.

Key features include:

- __Full Pandas API compatibility__ with a transparent fallback mechanism to native Pandas, ensuring that your workflows continue uninterrupted even if a feature is not yet supported.

- __Lazy evaluation of operations__, allowing powerful optimizations like filter pushdown and column pruning behind the scenes.

- __Scalable MPI-based execution__, leveraging HPC (High-Performance Computing) techniques for efficient parallelism; whether you're working on a laptop or running jobs across a large cloud cluster.

- __Vectorized execution__ with streaming and spill-to-disk capabilities, making it possible to process datasets larger than memory reliably.

!!! warning
    `bodo.pandas` is under active development and is currently considered experimental. Some features and APIs may not yet be fully supported. We welcome your feedback—please join our community Slack or open an issue on GitHub if you encounter any problems!

## Lazy Evaluation and Fallback to Pandas

`bodo.pandas` operates with lazy evaluation, meaning operations are recorded into a query plan rather than executed immediately. Execution is automatically triggered only when results are actually needed, such as when displaying the first 5 rows in a DataFrame with `print(df.head(5))`.

If the user code encounters an unsupported Pandas API or an unsupported parameter, `bodo.pandas` gracefully falls back to native Pandas. When this happens, the query plan is immediately executed, the resulting data is collected onto a single core and converted to a Pandas DataFrame, and further operations proceed using Pandas.

!!! warning
    Fallback to Pandas may lead to degraded performance and increase the risk of out-of-memory (OOM) errors, especially for large datasets.

--

## General Functions

### bodo.pandas.from_pandas
``` py
bodo.pandas.from_pandas(df: pandas.DataFrame) -> BodoDataFrame
```

Converts a Pandas DataFrame into a lazy BodoDataFrame.

#### Parameters
: __df__ : __*pandas.DataFrame*__: A Pandas DataFrame.


#### Returns
: __BodoDataFrame__: A lazy BodoDataFrame.

#### Examples

``` py
import pandas as pd
import bodo.pandas as bodo_pd

df = pd.DataFrame(
        {
            "a": [1, 2, 3, 7] * 2,
            "b": [4, 5, 6, 8] * 2,
            "c": ["a", "b", None, "abc"] * 2,
        },
    )

bdf = bodo_pd.from_pandas(df)
```

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

Creates a lazy DataFrame object for reading from parquet file(s).

#### Parameters
: __path__ : __*str, path object or file-like object*__: Location(s) of the parquet file(s). Refer to [Pandas documentation for more details](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html#pandas.read_parquet).

!!! warning
    The following parameters will trigger a fallback to [pandas.read_parquet](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html#pandas.read_parquet) if a value other than the default is provided: __engine__, __columns__, __storage_options__, __use_nullable_dtypes__, __dtype_backend__, __filesystem__, __filters__, __**kwargs__


#### Returns
: __BodoDataFrame__: A lazy BodoDataFrame.


## DataFrame API

### `apply(...) -> Series`

## Series API

### `map(args) -> Series`

### `str.strip(na_action : bool | None = None) -> Series`

### `str.lower() -> Series`




