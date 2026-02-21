---
hide:
  - footer
---

# Bodo Usage Guides {#guides}

This section provides a collection of guides to help you go from beginner to expert with Bodo. Choose a guide from the list below to get started.

---

## Installation and Setup Guides
This section provides guides to help you install and set up Bodo on your local machine or get onboarded on Bodo Platform on AWS or Azure.

- [Local Installation][install]
- [Bodo Platform on AWS][bodo_platform_aws]
- [Bodo Platform on Azure][bodo_platform_azure]

---

## [Bodo DataFrames Developer Guide][df_devguide]

This guide demonstrates how to use BodoDataFrames and will walk you through important concepts such as
lazy evaluation, optimizations and execution triggers, fallback to Pandas and user defined functions.

## [GPU Acceleration in Bodo DataFrames][df_gpu]

This guide discusses GPU acceleration of Bodo DataFrames and how this interacts with the CPU.  It describes how to enable GPU acceleration, certain environment variables useful for optimizing GPU execution, and which kinds of operations are supported or not supported on the GPU.

---

## [Iceberg][iceberg_intro]

This guide demonstrates how to use Bodo for reading and writing Iceberg tables.

---

## JIT Development Guides

---

### [Bodo JIT Dev Guide][jit_devguide]

If you are a developer and want to get started with Bodo JIT, this guide will help you get started with Bodo on your local machine.

### Understanding Parallelism with Bodo
This section provides a collection of guides to help you understand how Bodo parallelizes your code and how to write code that can benefit from Bodo's parallelism.

- [Basics of Bodo Parallelism][basics]
- [Advanced Parallelism Topics][advanced]
- [Typing Considerations][typing-considerations]
- [Unsupported Programs][notsupported]

### [Scalable Data I/O with Bodo][file_io]

This guide demonstrates how to use Bodo's file I/O APIs to read and write data.

### [Using Regular Python inside JIT with @bodo.wrap_python][objmode]

This guide teaches you how to can interleave regular Python code with Bodo functions using Bodo's object mode.

### [Measuring Performance][performance]

This guide provides an overview of how to correctly measure performance of your Bodo code.

### [Caching][caching]

This guide outlines how caching works in Bodo and best practices for using it.

### [Inlining][inlining]

This guide discusses advanced Bodo feature that allows you to inline functions to perform additional compiler optimizations.

### [Bodo Errors][bodoerrors]

This guide provides a list of common Bodo errors and tips on how to resolve them.

### [Compilation Tips][compilation]

This guide provides a list of tips to help you optimize your Bodo code for compilation and get the most out of Bodo.

### [Verbose Mode][bodoverbosemode]

For advanced developers, this guide provides an overview of Bodo's verbose mode and how to use it to debug your code.

---

## [Deploying Bodo with Kubernetes][kubernetes]

This guide walks through an example showing how to deploy a Bodo application with Kubernetes.

---

## Bodo Cloud Platform

This set of guides explains the basics of using the Bodo cloud platform and associated concepts.

- [Organization Basics][organization-basics]
- [Creating a Cluster][creating_clusters]
- [Using Notebooks][notebooks]
- [Running Jobs][running-jobs]
- [Native SQL with Catalogs][sql_catalog]
- [Instance Role for a Cluster][instance_role_cluster]
- [Managing Packages on the cluster using Jupyter magics - Conda and Pip][managing-packages]
- [Running shell commands on the cluster using Jupyter magics][shell-commands]
- [Connecting to a Cluster][connecting_to_a_cluster]
- [Troubleshooting][troubleshooting]

---

Don't see what you're looking for? Check out the [Bodo API Reference][apireference] for more information.
