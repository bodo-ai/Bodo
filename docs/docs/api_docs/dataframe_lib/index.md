# Bodo DataFrames API {#dataframe-lib}

Bodo DataFrames is designed to accelerate and scale Pandas workflows with just a one-line change — simply replace:

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
    Bodo DataFrames is under active development and is currently considered experimental.
    Some features and APIs may not yet be fully supported.
    We welcome your feedback — please join our community [Slack](https://bodocommunity.slack.com/join/shared_invite/zt-qwdc8fad-6rZ8a1RmkkJ6eOX1X__knA#/shared-invite/email) or open an issue on [our GitHub](https://github.com/bodo-ai/Bodo)
    if you encounter any problems!

## Lazy Evaluation and Fallback to Pandas

Bodo DataFrames operates with lazy evaluation to allow query optimization, meaning operations are recorded into a query plan rather than executed immediately.
Execution is automatically triggered only when results are actually needed, such as when displaying a DataFrame `df` with `print(df)`.

If the user code encounters an unsupported Pandas API or an unsupported parameter, Bodo DataFrames gracefully falls back to native Pandas.
When this happens, the current query plan of the DataFrame is immediately executed, the resulting data is collected onto a single core and converted to a Pandas DataFrame, and further operations proceed using Pandas.

!!! warning
    Fallback to Pandas may lead to degraded performance and increase the risk of out-of-memory (OOM) errors, especially for large datasets.

## GPU Acceleration

GPU acceleration for Bodo DataFrames is currently under development.  See [here][gpu] for more details.

[gpu]: ../../guides/dataframes/gpu_acceleration.md

<div class="grid cards" markdown>

- [General Functions][general-functions]
- [Dataframe API][dataframe]
- [Input/Output][inout]
- [Series API][series]
- [GroupBy][groupby]
- [AI Integration][ai]

</div>


[general-functions]: ../dataframe_lib/general_functions/index.md
[dataframe]: ../dataframe_lib/dataframe/index.md
[series]: ../dataframe_lib/series/index.md
[inout]: ../dataframe_lib/io.md
[groupby]: ../dataframe_lib/groupby/index.md
[ai]: ../dataframe_lib/ai.md
