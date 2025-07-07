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

# Bodo: High-Performance Python Compute Engine for Data and AI

Bodo is a cutting edge compute engine for large scale Python data processing. Powered by an innovative auto-parallelizing just-in-time compiler, Bodo transforms Python programs into highly optimized, parallel binaries without requiring code rewrites, which makes Bodo [20x to 240x faster](https://github.com/bodo-ai/Bodo/tree/main/benchmarks/nyc_taxi) compared to alternatives!

<img src="benchmarks/img/nyc-taxi-benchmark.png" alt="NYC Taxi Benchmark" width="500"/>

Unlike traditional distributed computing frameworks, Bodo:
- Seamlessly supports native Python APIs like Pandas and NumPy.
- Eliminates runtime overheads common in driver-executor models by leveraging Message Passing Interface (MPI) tech for true distributed execution.

## Goals

Bodo makes Python run much (much!) faster than it normally does!

1. **Exceptional Performance:**
Deliver HPC-grade performance and scalability for Python data workloads as if the code was written in C++/MPI, whether running on a laptop or across large cloud clusters.

2. **Easy to Use:**
Easily integrate into Python workflows with a simple decorator, and support native Pandas and NumPy APIs.

3. **Interoperable:**
Compatible with regular Python ecosystem, and can selectively speed up only the functions that are Bodo supported.

4. **Integration with Modern Data Infrastructure:**
Provide robust support for industry-leading data platforms like Apache Iceberg and Snowflake, enabling smooth interoperability with existing ecosystems.


## Non-goals

1. *Full Python Language Support:*
We are currently focused on a targeted subset of Python used for data-intensive and computationally heavy workloads, rather than supporting the entire Python syntax and all library APIs.

2. *Non-Data Workloads:*
Prioritize applications in data engineering, data science, and AI/ML. Bodo is not designed for general-purpose use cases that are non-data-centric.

3. *Real-time Compilation:*
While compilation time is improving, Bodo is not yet optimized for scenarios requiring very short compilation times (e.g., workloads with execution times of only a few seconds).


## Key Features

- Automatic optimization & parallelization of Python programs using Pandas and NumPy.
- Linear scalability from laptops to large-scale clusters and supercomputers.
- Advanced scalable I/O support for Iceberg, Snowflake, Parquet, CSV, and JSON with automatic filter pushdown and column pruning for optimized data access.
- High performance SQL Engine that is natively integrated into Python.

See Bodo documentation to learn more: https://docs.bodo.ai/


## Installation

Note: Bodo requires Python 3.9+.

Bodo can be installed using Pip or Conda:

```bash
pip install -U bodo
```

or

```bash
conda create -n Bodo python=3.13 -c conda-forge
conda activate Bodo
conda install bodo -c conda-forge
```

Bodo works with Linux x86, both Mac x86 and Mac ARM, and Windows right now. We will have Linux ARM support (and more) coming soon!

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
    df2 = pd.DataFrame({"A": df.apply(lambda r: 0 if r.A == 0 else (r.B // r.A), axis=1)})
    df2.to_parquet("out.pq")
    print("Execution time:", time.time() - t1)

computation()
```

## How to Contribute

Please read our latest [project contribution guide](CONTRIBUTING.md).

## Getting involved

You can join our community and collaborate with other contributors by joining our [Slack channel](https://bodocommunity.slack.com/join/shared_invite/zt-qwdc8fad-6rZ8a1RmkkJ6eOX1X__knA#/shared-invite/email) – we’re excited to hear your ideas and help you get started!

[![codecov](https://codecov.io/github/bodo-ai/Bodo/graph/badge.svg?token=zYHQy0R9ck)](https://codecov.io/github/bodo-ai/Bodo)