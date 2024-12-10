# Bodo Python Quick Start {#quickstart-local-python}

This quickstart guide will walk you through the process of running a simple Python computation using Bodo on your local machine.

## Prerequisites

[Install Bodo](../installation_and_setup/install.md) to get started (e.g., `pip install bodo` or `conda install bodo -c bodo.ai -c conda-forge`).

## Generate Sample Data

Let's start by creating a Parquet file with some sample data. The following Python code creates a Parquet file with two columns `A` and `B` and 20 million rows. The column `A` contains values from 0 to 29, and the column `B` contains values from 0 to 19,999,999.

```python
import pandas as pd
import numpy as np
import bodo
import time

NUM_GROUPS = 30
NUM_ROWS = 20_000_000
df = pd.DataFrame({
    "A": np.arange(NUM_ROWS) % NUM_GROUPS,
    "B": np.arange(NUM_ROWS)
})
df.to_parquet("my_data.pq")
```

## A Simple Pandas Computation

Now let's write a simple Python function that computes the sum of column `A` for all rows where `B` is greater than 4 using pandas. We decorate the function with `@bodo.jit` to indicate that we want to compile the code using Bodo. Let's also add a timer to measure the execution time.


```python
@bodo.jit(cache=True)
def computation():
    t1 = time.time()
    df = pd.read_parquet("my_data.pq")
    df1 = df[df.B > 4].A.sum()
    print("Execution time:", time.time() - t1)
    return df1

result = computation()
print(result)
```

## Running the Code

Bringing it all together, the complete code looks like this:

```python
import pandas as pd
import numpy as np
import bodo
import time

NUM_GROUPS = 30
NUM_ROWS = 20_000_000

df = pd.DataFrame({
    "A": np.arange(NUM_ROWS) % NUM_GROUPS,
    "B": np.arange(NUM_ROWS)
})
df.to_parquet("my_data.pq")

@bodo.jit(cache=True)
def computation():
    t1 = time.time()
    df = pd.read_parquet("my_data.pq")
    df1 = df[df.B > 4].A.sum()
    print("Execution time:", time.time() - t1)
    return df1

result = computation()
print(result)
```

To run the code, save it to a file, e.g. `test_bodo.py`, and run the following command in your terminal:

```bash
python test_bodo.py
```

By default Bodo will use all available cores. To set a limit on the number of processes spawned, set the environment variable `BODO_NUM_WORKERS`.
Note that the first time you run this code, it may take a few seconds to compile the code.
Next time you run the code, it will execute much faster. Check the [Python API Reference][pythonreference] for the full list of supported Python operations.
