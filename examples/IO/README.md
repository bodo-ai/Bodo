# Introduction

This folder contains examples of Bodo's IO capabilities. 

### 1. Optimizing  Snowflake retrieval times enabling faster I/O times.

- **File**: `read_snowflake_pandas.py` 

- **Description:** This notebook demonstrates reading data from Snowflake, a cloud data warehouse, using Bodo and Pandas. It showcases Bodo's ability to perform predicate pushdown and column pruning optimizations for efficient data loading.

- **Bodo Benefits:** It shows how Bodo accelerates data processing without requiring code changes. By leveraging predicate pushdown and column pruning, Bodo optimizes Snowflake queries to reduce I/O overhead and improve efficiency. It seamlessly distributes computations across multiple CPU cores, making large-scale data transformations and analytics much faster than standard Pandas.

### 2. Using Iceberg to reduce I/O Costs  across different data catalogs.

- **File**: `read_iceberg_sample.py` 

- **Description:** This demonstrates how Bodo can be used to read and write Iceberg tables, leveraging different data catalogs.

- **Bodo Benefits:** This gives a concrete example of how Bodo can be used to perform file I/O, and its compatibility with the increasingly popular Iceberg format.
