---
hide:
  - toc
  - footer
---

# Bodo: Simple Python Acceleration and Scaling for Data and AI


Bodo Dataframes is a drop in replacement for pandas (`import bodo.pandas as pd`, a single line import change),
which applies advanced High-Performance Computing (HPC) and database optimization
techniques to significantly boost the performance of your existing Python code without code rewrites.
Bodo also scales Python code from single laptops to large clusters and supercomputers automatically.

In addition, Bodo's auto-parallelizing
just-in-time (JIT) compiler transforms custom Python code using Pandas and Numpy into highly optimized,
parallel binaries without requiring code rewrites.


## Technical Differentiation

Unlike traditional distributed computing frameworks, Bodo:

- Provides drop-in replacement for Pandas APIs.
- Automatically optimizes queries at database-grade levels and code at HPC compiler levels for maximum efficiency.
- Eliminates overheads common in driver-executor models by leveraging Message Passing Interface (MPI) for true distributed execution.


## Key Features

- Automatic Optimization & Parallelization: Converts standard Python programs using Pandas and NumPy into high-performance parallel binaries automatically.
- Linear Scalability: Effortlessly scales from single laptops to large clusters and supercomputers.
- Optimized I/O Operations: Advanced data access capabilities for Iceberg, Snowflake, Parquet, CSV, and JSON with automatic optimizations like filter pushdown and column pruning.
- Integrated SQL Engine: Built-in, high-performance SQL capabilities directly within Python workflows.
