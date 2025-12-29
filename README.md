<!--
NOTE: the example in this file is covered by tests in bodo/tests/test_quickstart_docs.py. Any changes to the examples in this file should also update the corresponding unit test.
 -->

![Logo](Assets/bodo.png)

<h3 align="center">
  <a href="https://docs.bodo.ai/latest/" target="_blank"><b>Docs</b></a>
  &nbsp;&#183;&nbsp;
  <a href="https://bodocommunity.slack.com/join/shared_invite/zt-qwdc8fad-6rZ8a1RmkkJ6eOX1X__knA#/shared-invite/email" target="_blank"><b>Slack</b></a>
  &nbsp;&#183;&nbsp;
  <a href="https://www.bodo.ai/benchmarks/" target="_blank"><b>Benchmarks</b></a>
</h3>

# Bodo DataFrames: Drop-in Pandas Replacement for Acceleration and Scaling of Data and AI

Bodo DataFrames is a high performance DataFrame library for large scale Python data processing, AI/ML use cases.
It functions as a drop-in replacement for Pandas while providing additional Pandas-compatible APIs for simplifying and scaling AI workloads,
a just-in-time (JIT) compiler for accelerating custom transformations, as well as an integrated SQL engine for extra flexibility.

Under the hood, Bodo DataFrames relies on MPI-based high-performance computing (HPC) technology,
often making it orders of magnitude faster than tools like Spark or Dask.
Refer to our [NYC Taxi benchmark](https://github.com/bodo-ai/Bodo/tree/main/benchmarks/nyc_taxi)
for an example where Bodo is 2-240x faster than other systems:

<!-- TODO: updated graph with Taxi benchmark including Bodo DataFrames Pandas API -->
<img src="benchmarks/img/nyc-taxi-benchmark.png" alt="NYC Taxi Benchmark" width="500"/>

Unlike traditional distributed computing frameworks, Bodo DataFrames:
- Automatically scales and accelerates Pandas workloads with a single line of code change.
- Eliminates runtime overheads common in driver-executor models by leveraging Message Passing Interface (MPI) technology for true parallel execution.


## Goals

Bodo DataFrames makes Python run much (much!) faster than it normally does!

1. **Exceptional Performance:**
Deliver HPC-grade performance and scalability for Python data workloads as if the code was written in C++/MPI, whether running on a laptop or across large cloud clusters.

2. **Easy to Use:**
Easily integrate into Python workflows— it's as simple as changing `import pandas as pd` to `import bodo.pandas as pd`.

3. **Interoperable:**
Compatible with regular Python ecosystem, and can selectively speed up only the sections of the workload that are Bodo supported.

4. **Integration with Modern Data Infrastructure:**
Provide robust support for industry-leading data platforms like Apache Iceberg and Snowflake, enabling smooth interoperability with existing ecosystems.


## Key Features

- Drop-in Pandas replacement, (just change the import!) with a seamless fallback to vanilla Pandas to avoid breaking existing workloads.
- Intuitive APIs for simplifying and scaling AI workloads.
- Advanced query optimization,
C++ runtime,
and parallel execution using MPI to achieve the best possible performance while leveraging all available cores.
- Streaming execution to process larger-than-memory datasets.
- Just in time (JIT) compilation with native support for Pandas, Numpy and Scikit-learn
for accelerating custom transformations or performance-critical functions.
- High performance SQL engine that is natively integrated into Python.
- Advanced scalable I/O support for Iceberg, Snowflake, Parquet, CSV, and JSON with automatic filter pushdown and column pruning for optimized data access.

See Bodo DataFrames documentation to learn more: https://docs.bodo.ai/


## Installation

Note: Bodo DataFrames requires Python 3.10+.

Bodo DataFrames can be installed using Pip or Conda:

```bash
pip install -U bodo
```

or

```bash
conda create -n Bodo python=3.14 -c conda-forge
conda activate Bodo
conda install bodo -c conda-forge
```

Bodo DataFrames works with Linux x86, both Mac x86 and Mac ARM, and Windows right now. We will have Linux ARM support (and more) coming soon!

## Bodo DataFrames Example

Here is an example Pandas code that reads and processes a sample Parquet dataset.
Note that we replaced the typical import:
```python
import pandas as pd
```
with:
```python
import bodo.pandas as pd
```
which accelerates the following code segment by about 20-30x on a laptop.

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

## How to Contribute

Please read our latest [project contribution guide](CONTRIBUTING.md).

## Getting involved

You can join our community and collaborate with other contributors by joining our [Slack channel](https://bodocommunity.slack.com/join/shared_invite/zt-qwdc8fad-6rZ8a1RmkkJ6eOX1X__knA#/shared-invite/email) – we’re excited to hear your ideas and help you get started!

[![codecov](https://codecov.io/github/bodo-ai/Bodo/graph/badge.svg?token=zYHQy0R9ck)](https://codecov.io/github/bodo-ai/Bodo)