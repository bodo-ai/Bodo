GPU Acceleration for Bodo DataFrames {#df_gpu}
=================

This page describes Bodo’s CPU–GPU hybrid execution within [Bodo DataFrames][df_page] and uses terminology from that document, so it should be read first.  For hybrid execution, we discuss how GPU execution is enabled, what is supported today, configuration and tuning, and important caveats.

[df_page]: dataframes_intro.md

## Overview

Bodo DataFrames provides hybrid CPU-GPU execution.  It can execute anywhere from 0 to 100% of the nodes of a DataFrame plan on the GPUs available within the machine or a Bodo cluster.  Bodo DataFrames incorporates a cost model that analyzes the plan to determine which nodes should run on CPU or GPU.  When neighboring pipeline nodes run on different device types, Bodo automatically inserts and performs the necessary data transfers.

## Enabling GPU Hybrid Execution

GPU execution is opt-in. To enable GPU usage by the DataFrame system set:

```
export BODO_GPU=1
```

If BODO_GPU is not set (or set to 0), Bodo runs DataFrame execution on CPU only even if GPUs are present.

## How Placement is Decided

Currently, when GPU execution is enabled, Bodo will run every operation on the GPU for which we have a GPU implementation (see supported capabilities below).  However, this approach will be replaced in the near future with an advanced device placement algorithm that will use a cost-model to determine which plan nodes should run on CPU versus GPU such that the plan execution achieves the lowest latency.  In either case, when adjacent nodes in a plan are run on different device types, Bodo automatically inserts transfers between host and device as needed.

<!--
Bodo DataFrames uses a dynamic programming-based algorithm, whose goal is to minimize latency, in conjunction with a cost-model to determine which plan nodes should be run on CPU or GPU.  Currently, only some of the plan node types have a GPU implementation and can be run on GPU.  In addition, this algorithm uses the relative speed of the CPUs and GPUs in the system for each node type as well as the expected transfer times between CPU and GPU for each pair of plan nodes to determine which nodes should run on CPU or GPU.  When adjacent nodes in the plan are run on different device types, Bodo automatically inserts transfers between host and device as needed.  The first time BODO_GPU is enabled, Bodo will run a small number of example operators on CPU and GPU to determine their relative speeds and these statistics are later used to estimate execution time for operators on CPU and GPU.
-->

## Checking Placement

When GPU acceleration is enabled, users can view which plan nodes will be run on CPU versus GPU by setting the following environment variable:

```
export BODO_DATAFRAME_LIBRARY_DUMP_PLANS=1
```

The following is an example output when this environment variable is enabled.

```
┌───────────────────────────┐
│      (GPU) PROJECTION     │
│    ────────────────────   │
│        Expressions:       │
│           #[1.0]          │
│           #[1.1]          │
│           #[0.1]          │
│                           │
│          ~3 rows          │
└─────────────┬─────────────┘
┌─────────────┴─────────────┐
│   (GPU) COMPARISON_JOIN   │
│    ────────────────────   │
│      Join Type: INNER     │
│                           │
│        Conditions:        ├──────────────┐
│     (#[1.0] = #[0.0])     │              │
│                           │              │
│          ~3 rows          │              │
└─────────────┬─────────────┘              │
┌─────────────┴─────────────┐┌─────────────┴─────────────┐
│  (GPU) LogicalJoinFilter  ││(GPU) BODO_READ_PARQUE...  │
│    ────────────────────   ││    ────────────────────   │
│       filter_ids: 0       ││                           │
│                           ││                           │
│      filter_columns:      ││                           │
│          [[0], ]          ││                           │
│                           ││                           │
│    is_first_locations:    ││                           │
│         [[true], ]        ││                           │
│                           ││                           │
│    orig_build_key_cols:   ││                           │
│          [[0], ]          ││          ~3 rows          │
└─────────────┬─────────────┘└───────────────────────────┘
┌─────────────┴─────────────┐
│  (CPU) BODO_READ_DF(A, B) │
│    ────────────────────   │
│          ~3 rows          │
└───────────────────────────┘
```

## Configuration and Tuning

### Batch Size

GPUs generally prefer to work on larger chunks of data compared to CPU.  As such, Bodo has a separate GPU batch size that controls the size of batches flowing through the GPU with Bodo pipelines.  To set this batch size, use the following environment variable.

```
export BODO_GPU_STREAMING_BATCH_SIZE=320000   # default: 320K
```

Tune this value for your workload: larger batches increase GPU utilization but require more device memory.

### Memory Allocator

Because Bodo pipelines data in fixed-sized batches through the GPU, lots of room exists to improve performance by re-using memory allocations.  Therefore, we suggest that users enable the RMM pooling (or arena) allocator.  For example:

```
export RMM_ALLOCATOR="pool"
export RMM_POOL_INIT_SIZE="2GB"
```

Adjust RMM_POOL_INIT_SIZE to match your workload and available GPU memory.

## Supported Capabilities and Caveats

Below is a concise summary of broad capabilities that can run on GPU today, followed by specific caveats that may prevent a particular use of that capability from running on GPU.

* Parquet read (local filesystem, S3, HDFS, Azure Data Lake, Google Cloud Storage)

* Parquet write (local filesystem, S3, HDFS, Azure Data Lake, Google Cloud Storage)

* Row filtering (Pandas-style boolean filters) — UDFs excluded

* Column selection and vectorized arithmetic / boolean ops — UDFs excluded

* GroupBy aggregations: sum, count, mean, min, max, var, std, size, skew, nunique

* Inner joins with equality conditions.

* drop_duplicates

## Unsupported Capabilities

No other input types (Pandas dataframe, CSV, remote Iceberg reads, etc.) are currently supported on GPU. Those reads run on CPU.

Limit, sampling, CTEs, sorting, quantiles, and union are not currently supported.

## Important Per-Feature Caveats

### Read Parquet

A plan that includes a head() (or other operations that force a small sample collection via Pandas) may prevent the Parquet read from running on GPU; in such cases the read may fall back to CPU to satisfy the sampling semantics.

### Filtering

Vectorized boolean filters run on GPU. User-defined functions (UDFs) used inside filters are not supported on GPU and will force CPU execution for that node.

### Column expressions

Column selection and built-in arithmetic/boolean expressions are supported on GPU. UDFs are excluded.

### GroupBy

The listed aggregations (sum, count, mean, min, max, var, std, size, skew, nunique) are supported on GPU. Custom aggregations implemented as UDFs or Python callbacks will run on CPU.

### Joins

Inner equi-joins are supported on GPU. Joins with non-equality predicates (range joins, inequality joins, or arbitrary expressions) are not supported on GPU and will run on CPU.

## Troubleshooting

If execution is slower than expected, confirm the operators in your plan are supported on GPU (see supported list).

If GPU profiling shows high allocation overhead, ensure RMM pooling is enabled and tuned to an appropriate value.

If out of memory is reported on GPU, reduce BODO_GPU_STREAMING_BATCH_SIZE or increase RMM_POOL_INIT_SIZE if memory is available.

### Unexpected CPU fallback

If you expect a portion of your pipeline to run on GPU but it executes on CPU instead, check the following:

* Verify that the operators involved are listed as GPU-supported in the sections above; unsupported operators will always run on CPU.
* Ensure that GPU execution is enabled by setting `BODO_GPU=1` in the environment for the process running your code.
* Look for unsupported constructs such as UDFs inside filters, column expressions, or aggregations, which can force CPU execution for those nodes.
* Review your execution or plan diagnostics to confirm which nodes are placed on GPU vs CPU and adjust your code or configuration accordingly.

## Roadmap

Other operators and additional join variants are forthcoming. We are working on performance tuning and optimization as well.
