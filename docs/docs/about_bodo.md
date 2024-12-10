---
hide:
  - toc
  - footer
---

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
