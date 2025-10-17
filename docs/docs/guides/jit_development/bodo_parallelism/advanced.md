# Advanced Parallelism Topics {#advanced}

This page discusses parallelism topics that are useful for
performance tuning and advanced use cases.

## Distributed Flags For JIT Functions {#dist-flags}

Bodo infers data distributions for inputs and outputs of JIT functions automatically.
For example, all dataframe arguments and return values are distributed in this code:

```py
@bodo.jit
def f():
    df = pd.read_parquet("pd_example.pq")
    return df


@bodo.jit
def h(df):
    df2 = df.groupby("A").sum()
    return df2


@bodo.jit
def g(df):
    df3 = h(df)
    return df3


df = f()
df3 = g(df)
```

Bodo tracks distributions across JIT functions and between JIT
and regular Python code (by setting metadata in regular Pandas dataframes).
However, the user can specify distributions manunally as well.
The above code is equivalent to:

```py
@bodo.jit(distributed=["df"])
def f():
    df = pd.read_parquet("pd_example.pq")
    return df


@bodo.jit(distributed=["df", "df2"])
def h(df):
    df2 = df.groupby("A").sum()
    return df2


@bodo.jit(distributed=["df", "df3"])
def g(df):
    df3 = h(df)
    return df3


df = f()
df3 = g(df)
```

Generally, Bodo can handle distributions of most use cases automatically and
we do not recommend setting distributions manually due to the possibility of human error.
However, there are some advanced use cases where setting these flags may be desirable or necessary.
For example, when a small dataframe is an input to a join, setting its distribution to replicated can improve parallel performance. In the example below, a small dataframe `df2` is an argument to a join on a large dataframe `df1`, and we specify `df2` as replicated for better parallel performance.
```py
@bodo.jit(distributed=["df1"], replicated=["df2"])
def load_data():
    df1 = pd.read_parquet("my_large_data.pq")
    df2 = pd.read_parquet("my_tiny_data.pq")
    return df1, df2


@bodo.jit
def merge_data():
    df1, df2 = load_data()
    df3 = df1.merge(df2, on="id")
    df3.to_parquet("my_merged_data.pq")


merge_data()

```

Another potential use case is when we want to parallelize computation without distributing data,
for applications such as parameter tuning and simulations.
The example below creates some parameters, distributes them manually using `bodo.scatterv`,
and performs some computation on each one using a `bodo.prange` parallel loop.
The input dataframe `df` is replicated across processors since all of its values are needed
for computations on each parameter.
Functions `create_params` and `load_data` have `distributed=False` set,
which makes all of their data structures and computations replicated across processors.

!!! seealso "See Also"
    API Docs for [`bodo.scatterv`][bodoscatterv]

```py
@bodo.jit(distributed=False)
def create_params():
    params = [1, 3, 4, 5, 7, 8, 11, 15, 17, 21]
    params2 = [a * 2 for a in params]
    return np.array(params + params2)


@bodo.jit(distributed=False)
def load_data():
    df = pd.read_parquet("my_large_data.pq")
    return df


@bodo.jit
def run_params():
    params = create_params()
    df = load_data()
    params_dist = bodo.scatterv(params)
    n = len(params_dist)
    res = np.zeros(n)
    for i in bodo.prange(n):
        p = params_dist[i]
        res[i] = df.apply(lambda x, a: x.B % a, axis=1, a=p).sum()
    print(res.max())


run_params()
```

A similar flag is `distributed_block` which informs bodo that the data is
distributed in equal chunks across cores (as done and expected by Bodo).
Typically, this is used when output
of `bodo.scatterv` is passed to a JIT function to allow for optimization and parallelization of more complex code.
(This example assumes [SPMD launch mode](bodo_parallelism_basics.md#spmd))

```py
@bodo.jit(spawn=False, distributed_block=["A"])
def f(A):
    ...

data = bodo.scatterv(...)
f(data)
```

## Indexing Operations on Distributed Data

Distributed data is usually accessed and modified through high-level
Pandas and Numpy APIs. However, in many cases, Bodo allows indexing operations on distributed
data without code modification. Here are such cases that Bodo currently supports:

1.  Getting values using boolean array indexing, e.g. `B = A[A > 3]`.
    The output can be distributed, but may be imbalanced
    ([`bodo.rebalance()`](#load-balancing-distributed-data) can be used if necessary).

2.  Getting values using a slice, e.g. `B = A[::2]`. The output can be
    distributed, but may be imbalanced
    ([`bodo.rebalance()`](#load-balancing-distributed-data) can be used if necessary).

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
      df = pd.concat([df, pd.DataFrame({"A": np.arange(i)})], ignore_index=True)

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
      df = pd.concat([df, get_result(params[i])], ignore_index=True)

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
have cross-iteration dependencies except for supported reductions. Currently,
reductions using `+=`, `*=`, `min`, and `max` operators are supported.
Iterations are simply divided between processes and executed in parallel,
but reductions are handled using data exchange.

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

The user is also responsible for ensuring that control flow doesn't prevent
the loop from being reduced. This can occur when operations are potentially
applied unevenly or when the order the operation occurs in matters. This means
that mixing reductions and control flow breaks such as `break` or `raise` are
not supported.

The below example shows what happens when control flow prevents a reduction
from being parallelized:

``` py
import bodo
from bodo import prange
import numpy as np

@bodo.jit
def prange_test(n):
    A = np.random.ranf(n)
    s = 0
    for i in prange(len(A)):
        if A[i] % 2 == 0:
            s *= 2
        else:
            s += A[i]
    return s

res = prange_test(10)
print(res)
```

Output:

```console
numba.core.errors.UnsupportedRewriteError: Failed in bodo mode pipeline (step: convert to parfors)
Reduction variable s has multiple conflicting reduction operators.
```

## Integration with non-Bodo APIs

There are multiple methods for integration with APIs that Bodo does not
support natively:

1. Switch to [regular Python inside JIT functions with @bodo.wrap_python][objmode]
2. Pass data in and out of JIT functions

### Passing Distributed Data

By default, Bodo will transparently handle distributing inputs across all
processes and will lazily collect output back onto the main process as the data
is accessed. In other words, programs that access data outside of a JIT context
will incur some overhead as the data is collected back onto a single process,
while programs that pass data between JIT functions will run faster. Note that
peeking at the first few rows of data will also be fast and efficient but
operations that require the full table (e.g. printing out the entire table) will
trigger collection of values.

### Passing Distributed Data in SPMD launch mode

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

![](../../../img/advanced_parallelism_dataframe.svg#center)

## Run code on a single rank {#run_on_single_rank}

By default, all non-JIT code will only be run on a single rank. Within a JIT
function, if there's some code you want to only run from a single rank, you can
do so as follows:
```py
@bodo.wrap_python(bodo.types.none)
def rm_dir():
    # Remove directory
    import os, shutil
    if os.path.exists("data/data.pq"):
        shutil.rmtree("data/data.pq")


@bodo.jit
def f():
    if bodo.get_rank() == 0:
        rm_dir()

    # To synchronize all ranks before proceeding
    bodo.barrier()

    ...
```

This is similar in SPMD launch mode (where the whole script is launched as parallel
MPI processes), except you will need to ensure that code that must only run on a
single rank is protected even outside of JIT functions:

```py
if bodo.get_rank() == 0:
    # Remove directory
    import os, shutil
    if os.path.exists("data/data.pq"):
        shutil.rmtree("data/data.pq")

# To synchronize all ranks before proceeding
bodo.barrier()
```

## Run code once on each node {#run_on_each_node}

In cases where some code needs to be run once on each node in a
multi-node cluster, such as a file system operation, installing
packages, etc., it can be done as follows from inside a JIT function:

```py
if bodo.get_rank() in bodo.get_nodes_first_ranks():
    # Remove directory on all nodes
    import os, shutil
    if os.path.exists("data/data.pq"):
        shutil.rmtree("data/data.pq")

# To synchronize all ranks before proceeding
bodo.barrier()
```

In SPMD launch mode the above can also be run outside of JIT functions.


!!! warning
    Running code on a single rank or a subset of ranks can lead to
    deadlocks. Ensure that your code doesn't include any MPI or Bodo
    functions.
