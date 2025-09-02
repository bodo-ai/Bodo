<!--
NOTE: the examples in this file are covered by tests in bodo/tests/test_quickstart_docs.py. Any changes to examples in this file should also update the corresponding unit test(s).
 -->

# Bodo Python Quick Start {#quickstart-local-python}

This quickstart guide will walk you through the process of running a simple Python computation using Bodo on your local machine.

## Installation

[Install Bodo](../installation_and_setup/install.md) to get started (e.g., `pip install -U bodo` or `conda install bodo -c conda-forge`).


## Drop-in Pandas Replacement with Bodo DataFrames

Bodo DataFrames can be used as a drop-in replacement for Pandas by changing `import pandas as pd` with `import bodo.pandas as pd`. For example:

```python
import bodo.pandas as pd
import numpy as np

n = 20_000_000
df = pd.DataFrame({"A": np.arange(n) % 30, "B": np.arange(n)})
df2 = df.groupby("A", as_index=False)["B"].max()
df2.to_parquet("my_data.pq")
```

Bodo DataFrames will optimize and parallelize the code automatically when possible.
It will fall back to Pandas seamlessly when some API isn't supported yet and throw a warning.
See the [Bodo DataFrames API reference][dataframe-lib] for supported Pandas APIs.


## Bodo JIT Compilation for Custom Code

JIT compilation converts Python functions to optimized parallel binaries,
which can provide orders of magnitude performance boost for custom code
like user defined functions (UDFs). For example:

```python
import bodo
import bodo.pandas as pd
import numpy as np

@bodo.jit
def f(df):
    return df.apply(lambda r: 0 if r.A == 0 else (r.B // r.A), axis=1)

n = 20_000_000
df = pd.DataFrame({"A": np.arange(n) % 30, "B": np.arange(n)})
S = f(df)
pd.DataFrame({"C": S}).to_parquet("my_data.pq")
```

All the code in JIT functions has to be compilable by Bodo JIT (will throw appropriate errors otherwise).
See [JIT development guide][devguide] and [JIT API reference][pythonreference] for supported Python features and APIs.
