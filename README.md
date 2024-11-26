# Bodo: High-Performance Python Compute Engine for Data and AI

Bodo is a cutting-edge compute engine that brings high-performance computing (HPC) speed
and scalability to Python data and AI programs. Powered by an innovative auto-parallelizing
just-in-time (JIT) compiler, Bodo transforms Python programs into highly optimized,
parallel binaries without requiring code rewrites.

Unlike traditional distributed computing frameworks, Bodo:
- Seamlessly supports native Python APIs like Pandas and NumPy.
- Eliminates runtime overheads common in driver-executor models by leveraging Message Passing Interface (MPI) technology for true distributed execution.


## Key Features

- Automatic optimization & parallelization of Python programs using Pandas and NumPy.
- Linear scalability from laptops to large-scale clusters and supercomputers.
- Advanced scalable I/O support for Iceberg, Snowflake, Parquet, CSV, and JSON with automatic filter pushdown and column pruning for optimized data access.
- High-Performance SQL Engine that is natively integrated into Python.

See Bodo documentation to learn more: https://docs.bodo.ai/


## Installation

Bodo can be installed using Pip or Conda:

```bash
pip install -U bodo
```

or 

```bash
conda create -n Bodo python=3.12 -c conda-forge
conda activate Bodo
conda install bodo -c bodo.ai -c conda-forge
```

## Example Code

Here is an example Pandas code that reads and processes a sample Parquet dataset with Bodo.


```python
import pandas as pd
import numpy as np
import bodo
import time

# Generate sample data
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

## How to Contribute

Please read our latest [project contribution guide](CONTRIBUTING.md).

## Getting involved

You can join our community and collaborate with other contributors by joining our [Slack channel](https://bodocommunity.slack.com/join/shared_invite/zt-qwdc8fad-6rZ8a1RmkkJ6eOX1X__knA#/shared-invite/email) – we’re excited to hear your ideas and help you get started!
