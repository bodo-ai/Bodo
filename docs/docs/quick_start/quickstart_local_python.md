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
import time

NUM_GROUPS = 30
NUM_ROWS = 20_000_000

df = pd.DataFrame({
    "A": np.arange(NUM_ROWS) % NUM_GROUPS,
    "B": np.arange(NUM_ROWS)
})
df.to_parquet("my_data.pq")

def computation():
    t1 = time.time()
    df = pd.read_parquet("my_data.pq")
    df["C"] = df.apply(lambda r: 0 if r.A == 0 else (r.B // r.A), axis=1)
    df.to_parquet("out.pq")
    print("Execution time:", time.time() - t1)

computation()
```

Bodo DataFrames will optimize and parallelize the code automatically when possible.
It will fall back to Pandas seamlessly when some API isn't supported yet and throw a warning.
See the [Bodo DataFrames API reference][dataframe-lib] for supported Pandas APIs.


## Bodo JIT Compilation for Best Native End-to-end Performance

JIT compilation converts Python functions to optimized parallel binaries.
Unlike Bodo DataFrames, JIT can optimize both Pandas and Numpy operations together and
in some cases provide better performance over Bodo DataFrames.
For example:

```python
import bodo
import pandas as pd
import numpy as np
import time

NUM_GROUPS = 30
NUM_ROWS = 20_000_000

df = pd.DataFrame({
    "A": np.arange(NUM_ROWS) % NUM_GROUPS,
    "B": np.arange(NUM_ROWS)
})

@bodo.jit
def computation(df):
    t1 = time.time()
    df["C"] = df.apply(lambda r: 0 if r.A == 0 else (r.B // r.A), axis=1)
    df["D"] = np.sin(df.A)
    df.to_parquet("out.pq")
    print("Execution time:", time.time() - t1)

computation(df)
```

All the code in JIT functions has to be compilable by Bodo JIT (will throw appropriate errors otherwise).
See [JIT development guide][devguide] and [JIT API reference][pythonreference] for supported Python features and APIs.
