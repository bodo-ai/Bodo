# Advanced Parallelism Topics {#advanced}

This section discusses parallelism topics that may be useful for
performance tuning and advanced use cases.

## Getting/Setting Distributed Data Directly

Distributed data is usually accessed and modified through high-level
Pandas and Numpy APIs. However, in many cases, Bodo allows direct access to distributed
data without code modification. Here are such cases that Bodo currently supports:

1.  Getting values using boolean array indexing, e.g. `B = A[A > 3]`.
    The output can be distributed, but may be imbalanced
    (`bodo.rebalance()` can be used if necessary).
    
[todo]: <> (add a reference to bodo.rebalance)

2.  Getting values using a slice, e.g. `B = A[::2]`. The output can be
    distributed, but may be imbalanced 
    (`bodo.rebalance()` can be used if necessary).
    
[todo]: <> (add a reference to bodo.rebalance)

3.  Getting a value using a scalar index, e.g. `a = A[m]`. The output
    can be replicated.
    
4.  Setting values using boolean array indexing, e.g. `A[A > 3] = a`.
    Only supports setting a scalar or lower-dimension value currently.
    
5.  Setting values using a slice, e.g. `A[::2] = a`. Only supports
    setting a scalar or lower-dimension value currently.
    
6.  Setting a value using a scalar index, e.g. `A[m] = a`.

## Concatenation Reduction

Some algorithms require generating variable-length output data per input
data element. Bodo supports parallelizing this pattern, which we refer
to as *concatenation reduction*. For example:

```py
@bodo.jit
def impl(n):
   df = pd.DataFrame()
   for i in bodo.prange(n):
      df = df.append(pd.DataFrame({"A": np.arange(i)}))

   return df
```

A common use case is simulation applications that generate possible
outcomes based on parameters. For example:

```py
@bodo.jit
def impl():
   params = np.array([0.1, 0.2, 0.5, 1.0, 1.2, 1.5, ..., 100])
   params = bodo.scatterv(params)
   df = pd.DataFrame()
   for i in bodo.prange(len(params)):
      df = df.append(get_result(params[i]))

   return df
```

In this example, we chose to manually parallelize the parameter array
for simplicity, since the workload is compute-heavy and the parameter
data is relatively small.

## Load Balancing Distributed Data

Some computations such as `filter`, `join` or `groupby` can result in
imbalanced data chunks across cores for distributed data. This may
result in some cores operating on nearly empty dataframes, and others on
relatively large ones.

Bodo provides `bodo.rebalance` to allow manual load balance if
necessary. For example:

```
@bodo.jit(distributed={"df"})
def rebalance_example(df):
    df = df[df["A"] > 3]
    df = bodo.rebalance(df)
    return df.sum()
```

In this case, we use `bodo.rebalance` to make sure the filtered
dataframe has near-equal data chunk sizes across cores, which would
accelerate later computations (`sum` in this case).

We can also use the `dests` keyword to specify a subset of ranks to
which bodo should distribute the data from all ranks.

Example usage:

```py
@bodo.jit(distributed={"df"})
def rebalance_example(df):
    df = df[df["A"] > 3]
    df = bodo.rebalance(df, dests=[0, 1])
    return df.sum()
```

## Explicit Parallel Loops

Sometimes explicit parallel loops are required since a program cannot be
written in terms of data-parallel operators easily. In this case, one
can use Bodo's `prange` in place of `range` to specify that a loop can
be parallelized. The user is required to make sure the loop does not
have cross iteration dependencies except for supported reductions.

The example below demonstrates a parallel loop with a reduction:

``` py
import bodo
from bodo import prange
import numpy as np

@bodo.jit
def prange_test(n):
    A = np.random.ranf(n)
    s = 0
    B = np.empty(n)
    for i in prange(len(A)):
        bodo.parallel_print("rank", bodo.get_rank())
        # A[i]: distributed data access with loop index
        # s: a supported sum reduction
        s += A[i]
        # write array with loop index
        B[i] = 2 * A[i]
    return s + B.sum()

res = prange_test(10)
print(res)
```

Output: 

```console
[stdout:0]
rank 0
rank 0
rank 0
13.077183553245497
[stdout:1]
rank 1
rank 1
rank 1
13.077183553245497
[stdout:2]
rank 2
rank 2
13.077183553245497
[stdout:3]
rank 3
rank 3
13.077183553245497
```

Currently, reductions using `+=`, `*=`, `min`, and `max` operators are
supported. Iterations are simply divided between processes and executed
in parallel, but reductions are handled using data exchange.

## Integration with non-Bodo APIs

There are multiple methods for integration with APIs that Bodo does not
support natively: 
1. Switch to python object mode inside jit functions
2. Pass data in and out of jit functions

### Passing Distributed Data

Bodo can receive or return chunks of distributed data to allow flexible
integration with any non-Bodo Python code. The following example passes
chunks of data to interpolate with Scipy, and returns interpolation
results back to jit function.

``` py
import scipy.interpolate

@bodo.jit(distributed=["X", "Y", "X2"])
def dist_pass_test(n):
    X = np.arange(n)
    Y = np.exp(-X/3.0)
    X2 = np.arange(0, n, 0.5)
    return X, Y, X2

X, Y, X2 = dist_pass_test(100)
# clip potential out-of-range values
X2 = np.minimum(np.maximum(X2, X[0]), X[-1])
f = scipy.interpolate.interp1d(X, Y)
Y2 = f(X2)

@bodo.jit(distributed={"Y2"})
def dist_pass_res(Y2):
    return Y2.sum()

res = dist_pass_res(Y2)
print(res)
```

```console
[stdout:0] 6.555500504321469 
[stdout:1] 6.555500504321469
[stdout:2] 6.555500504321469 
[stdout:3] 6.555500504321469
```

## Collections of Distributed Data

List and dictionary collections can be used to hold distributed data
structures:

``` py
@bodo.jit(distributed=["df"])
def f():
    to_concat = []
    for i in range(10):
        to_concat.append(pd.DataFrame({'A': np.arange(100), 'B': np.random.random(100)}))
        df = pd.concat(to_concat)
    return df

f()
```

![](../img/advanced_parallelism_dataframe.svg#center)

## Run code on a single rank {#run_on_single_rank}

In cases where some code needs to be run on a single MPI rank, you can
do so in a python script as follows:

```py
if bodo.get_rank() == 0:
    # Remove directory
    import os, shutil
    if os.path.exists("data/data.pq"):
        shutil.rmtree("data/data.pq")

# To synchronize all ranks before proceeding
bodo.barrier()
```

When running code on an IPyParallel cluster using the `%%px` magic, you
can do this instead:

``` py
%%px --targets 0
# Install package
!conda install pandas-datareader
```

An alias can be defined for convenience:

``` py
%alias_magic p0 px -p "--targets 0"
```

This can be used as any other magic:

``` py
%%p0
# Install package
!conda install pandas-datareader
```

## Run code once on each node {#run_on_each_node}

In cases where some code needs to be run once on each node in a
multi-node cluster, such as a file system operation, installing
packages, etc., it can be done as follows:

```py
if bodo.get_rank() in bodo.get_nodes_first_ranks():
    # Remove directory on all nodes
    import os, shutil
    if os.path.exists("data/data.pq"):
        shutil.rmtree("data/data.pq")

# To synchronize all ranks before proceeding
bodo.barrier()
```

The same can be done when running on an IPyParallel cluster using the
`%%px` magic:

``` py
%%px
if bodo.get_rank() in bodo.get_nodes_first_ranks():
    # Install package on all nodes
    !conda install pandas-datareader
```

!!! warning
    Running code on a single rank or a subset of ranks can lead to
    deadlocks. Ensure that your code doesn't include any MPI or Bodo
    functions.

