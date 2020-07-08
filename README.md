# Bodo Engine

[![Build Status](https://dev.azure.com/bodo-inc/Bodo/_apis/build/status/Bodo-inc.Bodo?branchName=master)](https://dev.azure.com/bodo-inc/Bodo/_build/latest?definitionId=1&branchName=master)
[![codecov](https://codecov.io/gh/Bodo-inc/Bodo/branch/master/graph/badge.svg?token=zYHQy0R9ck)](https://codecov.io/gh/Bodo-inc/Bodo)

## The Most Efficient Data Science Engine

Bodo scales analytics/ML codes in Python
to bare-metal cluster/cloud performance automatically.
It compiles a subset of Python (Pandas/Numpy) to efficient parallel binaries
with MPI, requiring only minimal code changes.
Bodo is orders of magnitude faster than
alternatives like [Apache Spark](http://spark.apache.org).

[Development guide](https://github.com/Bodo-inc/Bodo/tree/master/dev_docs/source) has these sections:
- [Getting Started](https://github.com/Bodo-inc/Bodo/blob/master/dev_docs/source/getting_started.rst)
- [Building Bodo from Source](https://github.com/Bodo-inc/Bodo/blob/master/dev_docs/source/build_bodo.rst)
- [Bodo Engine Development](https://github.com/Bodo-inc/Bodo/blob/master/dev_docs/source/bodo_dev.rst): Compiler Stages, Builtin Functions, IR Extensions.
- [Development Process](https://github.com/Bodo-inc/Bodo/blob/master/dev_docs/source/dev_process.rst): Test Suite, Code Structure, Debugging, Code Style, Code Coverage, DevOps, and Performance Benchmarking.
- [Github Practices](https://github.com/Bodo-inc/Bodo/blob/master/dev_docs/source/github_practices.rst)
- [Useful Numba knowledge](https://github.com/Bodo-inc/Bodo/blob/master/dev_docs/source/numba_info.rst)
- [Development using Docker](https://github.com/Bodo-inc/Bodo/blob/master/dev_docs/source/dev_with_docker.rst)
- [Conda Build](https://github.com/Bodo-inc/Bodo/blob/master/dev_docs/source/conda_build.rst)
- [Customer Code Rewrite](https://github.com/Bodo-inc/Bodo/blob/master/dev_docs/source/testing_poc_SQL.rst)
- [Release Checklist](https://github.com/Bodo-inc/Bodo/blob/master/dev_docs/source/release_checklist.rst)
	
To build the Development guide locally and view it with your browser, follow instructions [here](https://github.com/Bodo-inc/Bodo/blob/master/dev_docs/source/dev_process.rst#building-documentation)

Bodo Documentation: https://docs.bodo.ai
