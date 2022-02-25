# Bodo Parallelism Basics {#basics}

In this section, we will discuss Bodo's JIT compilation workflow and
the parallelism model and APIs provided by Bodo.

## JIT (Just-in-time) Compilation Workflow {#jit}

Bodo provides a just-in-time (JIT) compilation workflow using the
`@bodo.jit` decorator, which replaces a Python function with a so-called
`Dispatcher` object. Bodo compiles the function the first time a
Dispatcher object is called and reuses the compiled version afterwards.
The function is recompiled only if the same function is called with
different argument types (not often in practice). All of this is
completely transparent to the caller, and does not affect any Python
code calling the function.

```py 
>>> import numpy as np
>>> import pandas as pd
>>> import bodo
>>> @bodo.jit
... def f(n, a):
...   df = pd.DataFrame({"A": np.arange(n) + a})
...   return df.head(3)
... 
>>> print(f)
CPUDispatcher(<function f at 0x100bec310>)
>>> print(f(8, 1)) # compiles for (int, int) input types
   A
0  1
1  2
2  3
>>> print(f(8, 2)) # same input types, no need to compile
   A
0  2
1  3
2  4
>>> print(f(8, 2.2)) # compiles for (int, float) input types
     A
0  2.2
1  3.2
2  4.2
```

!!! note
    In many cases, the binary that Bodo generates when compiling a function
    can be saved to disk and reused across program executions. See
    [caching][caching] for more information.

## Parallel Execution Model

As we saw in the "Getting Started" tutorial, Bodo transforms functions
for parallel execution. Bodo uses Message Passing Interface
([MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)) that
follows Single Program Multiple Data
([SPMD](https://en.wikipedia.org/wiki/SPMD)) paradigm. In this model,
the dispatcher does not launch processes or threads on the fly. Instead,
all processes are launched at the beginning and run the same file using
`mpiexec` command.

Bodo parallelizes functions with the `bodo.jit` decorator by
distributing the data across the processes. Each rank runs the same code
on a chunk of the data, and Bodo automatically communicates the data
between the ranks (as needed).

For example, save the following code in a`test_bodo.py` and use
`mpiexec` to launch 4 processes as follows:

```py 
import numpy as np
import pandas as pd
import bodo


@bodo.jit
def f(n, a):
    df = pd.DataFrame({"A": np.arange(n) + a})
    return df


print(f(8, 1))
```

```shell 
mpiexec -n 4 python test_bodo.py
```

Output:

```console
   A
2  3
3  4
   A
6  7
7  8
   A
4  5
5  6
   A
0  1
1  2
```

In this example, `mpiexec` launches 4 Python processes, each 
executing the same `test_bodo.py` file. Since the function `f` is
decorated with `bodo.jit` and Bodo can parallelize it, each
process generates a chunk of the data in `np.arange`.

Note how the prints, which are regular Python code executed outside of
Bodo, run for each process.

!!! warning
    - Python codes outside of Bodo functions execute sequentially on
    every process.
    - Bodo functions run in parallel assuming that Bodo is
    able to parallelize them. Otherwise, Bodo prints the following warning
    and runs sequentially on every process.
    
    ```console
    BodoWarning: No parallelism found for function
    ```

On Jupyter notebook, parallel execution happens in very much the same
way. We start a set of MPI engines through `ipyparallel` and activate a
client. See [how to use bodo with jupyter notebooks][jupyter] for more information
and examples.

!!! seealso "See Also"
    [Parallel APIs][bodoparallelapis]

## Data Distribution

Bodo parallelizes computation by dividing data into separate chunks
across processes. However, some data handled by a Bodo function may not
be divided into chunks. There are are two main data distribution
schemes:

-   Replicated (*REP*): the data associated with the variable is the
    same on every process.
-   One-dimensional (*1D*): the data is divided into chunks, split along
    one dimension (rows of a dataframe or first dimension of an array).

Bodo determines distribution of variables automatically, using the
nature of the computation that produces them. Let's see an example:

```py
import bodo
import pandas as pd
@bodo.jit
def mean_power_speed():
    df = pd.read_parquet("data/cycling_dataset.pq")
    m = df[["power", "speed"]].mean()
    return m

res = mean_power_speed()
print(res)
```

Save code in mean_power_speed.py and run it with `mpiexec` as follows:

```shell
mpiexec -n 4 python mean_power_speed.py
```

```console
[stdout:0]
power    102.078421
speed      5.656851
dtype: float64
[stdout:1]
power    102.078421
speed      5.656851
dtype: float64
[stdout:2]
power    102.078421
speed      5.656851
dtype: float64
[stdout:3]
power    102.078421
speed      5.656851
dtype: float64
```

In this example, `df` is parallelized (each process reads a different
chunk) but `m` is replicated, even though it is a Series. Semantically,
it makes sense for the output of `mean` operation to be replicated on
all processors, since it is a reduction and produces "small" data.

### Distributed Diagnostics

The distributions found by Bodo can be printed either by setting the
environment variable `BODO_DISTRIBUTED_DIAGNOSTICS=1` or calling
`distributed_diagnostics()` on the compiled function. Let's examine the
previous example's distributions by adding following line to
`mean_power_speed` script:

```shell
mean_power_speed.distributed_diagnostics()
```

```shell
python mean_power_speed.py
```

```console
Distributed analysis replicated return variable $30return_value.12. Set distributed flag for the original variable if distributed partitions should be returned.
[stdout:0]
python mean_power_speed.py             
power    102.078421
speed      5.656851
dtype: float64
Distributed diagnostics for function mean_power_speed, /Users/mean_power_speed.py (3)

Data distributions:
    pq_table.0                                                              1D_Block
    pq_index.1                                                              1D_Block
    data_74                                                                 REP

    Parfor distributions:
       0                    1D_Block
       1                    1D_Block

    Distributed listing for function mean_power_speed, /Users/hadia/Bodo/testing/mean_power_speed.py (3)
    ---------------------------------------------------------------------| parfor_id/variable: distribution
    @bodo.jit                                                            | 
    def mean_power_speed():                                              | 
        df = pd.read_parquet("Bodo-tutorial/data/cycling_dataset.pq")----| pq_table.0: 1D_Block, pq_index.1: 1D_Block
        m = df[["power", "speed"]].mean()--------------------------------| #0: 1D_Block, #1: 1D_Block, data_74: REP
        return m                                                         | 

    Setting distribution of variable 'impl_v48_data_74' to REP: output of np.asarray() call on non-array is REP
```


Bodo compiler optimizations rename the variables. 
The output shows that `power` and `speed` columns of `df` are distributed (`1D_Block`), but `m` is replicated (`REP`). 
This is because `df` is the output from `read_parquet` and input to `mean`, both of which can be distributed by Bodo. 
`m` is the output from `mean`, which is replicated (available on every process).

[comment]: <> (Autorefs to [caching] and [jupyter] will be populated once those sections are added)
[todo]: <> (Modify/remove comment above as [caching] and [jupyter] sections are added)