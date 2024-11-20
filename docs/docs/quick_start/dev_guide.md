---
hide:
  - tags

tags:
  - getting started

---

# Python Development Guide {#devguide}


This page provides an introduction to Python programming with Bodo and explains its
important concepts briefly.

Installation
------------

We recommend Bodo Platform on AWS or Azure for using Bodo.
See how to [get started with the Bodo platform](quick_start_platform.md).
If you prefer a local environment, [Conda](https://docs.conda.io){target="blank"} is the recommended way to install Bodo locally. You can install the _Community Edition_ using conda, which allows you to use Bodo for free on up to 8 cores. 


```console 
conda create -n Bodo python=3.12 -c conda-forge
conda activate Bodo
conda install bodo -c bodo.ai -c conda-forge
```

These commands create a conda environment called `Bodo` and install Bodo Community Edition.

Data Transform Example with Bodo
--------------------------------

We use a simple data transformation example to discuss some of the key
Bodo concepts.

### Generate data

Let's generate some example data and write to a
[Parquet](http://parquet.apache.org/){target="blank"} file:

``` py
import pandas as pd
import numpy as np

# 10m data points
df = pd.DataFrame(
    {
        "A": np.repeat(pd.date_range("2013-01-03", periods=1000), 10_000),
        "B": np.arange(10_000_000),
    }
)
# set some values to NA
df.iloc[np.arange(1000) * 3, 0] = pd.NA
# using row_group_size helps with efficient parallel read of data later
df.to_parquet("pd_example.pq", row_group_size=100_000)
```

Save this code in `gen_data.py` and run in command line:

``` 
python gen_data.py
```

### Example Pandas Code {#example_code_in_pandas}

Here is a simple data transformation code in Pandas that processes a
column of datetime values and creates two new columns:

``` py
import pandas as pd
import time


def data_transform():
    t0 = time.time()
    df = pd.read_parquet("pd_example.pq")
    df["B"] = df.apply(lambda r: "NA" if pd.isna(r.A) else "P1" if r.A.month < 5 else "P2", axis=1)
    df["C"] = df.A.dt.month
    df.to_parquet("pandas_output.pq")
    print("Total time: {:.2f}".format(time.time() - t0))


if __name__ == "__main__":
    data_transform()
```

Save this code in `data_transform.py` and run in command line:

```console
$ python data_transform.py
Total time: 166.18
```

Standard Python is quite slow for these data transforms since:

1.  The use of custom code inside `apply()` does not let Pandas run an
    optimized prebuilt C library in its backend. Therefore, the Python
    interpreter overheads dominate.
2.  Python uses a single CPU core and does not parallelize
    computation.

Bodo solves both of these problems as we demonstrate below.

### Using the Bodo JIT Decorator

Bodo optimizes and parallelizes data workloads by providing just-in-time
(JIT) compilation. This code is identical to the original Pandas code,
except that it annotates the `data_transform` function with the
`bodo.jit` decorator:

``` py
import pandas as pd
import time
import bodo

@bodo.jit
def data_transform():
    t0 = time.time()
    df = pd.read_parquet("pd_example.pq")
    df["B"] = df.apply(lambda r: "NA" if pd.isna(r.A) else "P1" if r.A.month < 5 else "P2", axis=1)
    df["C"] = df.A.dt.month
    df.to_parquet("bodo_output.pq")
    print("Total time: {:.2f}".format(time.time()-t0))

if __name__ == "__main__":
    data_transform()
```

Save this code in `bodo_data_transform.py` and run on a single core from
command line:

``` 
$ BODO_NUM_WORKERS=1 python bodo_data_transform.py
Total time: 1.78
```

This code is *94x* faster with Bodo than Pandas even on a single core,
because Bodo compiles the function into a native binary, eliminating the
interpreter overheads in `apply`.

Now let's run the code on all CPU cores - the example below assumes an 8 core
machine.

``` 
$ python bodo_data_transform.py
Total time: 0.38
```

Using 8 cores gets an additional *~5x* speedup. The same program can be
scaled to larger datasets and as many cores as necessary in compute
clusters and cloud environments. An explicit limit on the number of cores used
can be set with the environment variable `BODO_NUM_WORKERS`, but by default Bodo
will use all available cores.

See the documentation on [bodo parallelism basics][basics] for more
details about Bodo's JIT compilation workflow and parallel computation
model.

### Compilation Time and Caching

Bodo's JIT workflow compiles the function the first time it is called,
but reuses the compiled version for subsequent calls. In the previous
code, we added timers inside the function to avoid measuring compilation
time. Let's move the timers outside and call the function twice:

``` py
import pandas as pd
import time
import bodo

@bodo.jit
def data_transform():
    df = pd.read_parquet("pd_example.pq")
    df["B"] = df.apply(lambda r: "NA" if pd.isna(r.A) else "P1" if r.A.month < 5 else "P2", axis=1)
    df["C"] = df.A.dt.month
    df.to_parquet("bodo_output.pq")

if __name__ == "__main__":
    t0 = time.time()
    data_transform()
    print("Total time first call: {:.2f}".format(time.time()-t0))
    t0 = time.time()
    data_transform()
    print("Total time second call: {:.2f}".format(time.time()-t0))
```

Save this code in `data_transform2.py` and run in command line:

``` 
$ python data_transform2.py
Total time first call: 4.72
Total time second call: 1.92
```

The first call is slower due to compilation of the function, but the
second call reuses the compiled version and runs faster.

Compilation time can be avoided across program runs by using the
`cache=True` flag:

``` py
import pandas as pd
import time
import bodo


@bodo.jit(cache=True)
def data_transform():
    df = pd.read_parquet("pd_example.pq")
    df["B"] = df.apply(lambda r: "NA" if pd.isna(r.A) else "P1" if r.A.month < 5 else "P2", axis=1)
    df["C"] = df.A.dt.month
    df.to_parquet("bodo_output.pq")


if __name__ == "__main__":
    t0 = time.time()
    data_transform()
    print("Total time: {:.2f}".format(time.time() - t0))
```

Save this code in `data_transform_cache.py` and run in command line
twice:

``` 
$ python data_transform_cache.py
Total time: 4.70
$ python data_transform_cache.py
Total time: 1.96
```

In this case, Bodo saves the compiled version of the function to a file
and reuses it in the second run since the code has not changed. We plan
to make caching default in the future. See [caching][caching] for more information.

Parallel Python Processes
-------------------------

Bodo will execute code decorated with `bodo.jit` in parallel. The function is
run on all cores, but Bodo divides the data and computation in JIT functions to
exploit parallelism.

Let's try a simple example that demonstrates how chunks of data are
loaded in parallel:

``` py
import pandas as pd
import bodo


def load_data_pandas():
    df = pd.read_parquet("pd_example.pq")
    print("pandas dataframe: ", df)


@bodo.jit
def load_data_bodo():
    df = pd.read_parquet("pd_example.pq")
    print("Bodo dataframe: ", df)


if __name__ == "__main__":
    load_data_pandas()
    load_data_bodo()
```

Save this code in `load_data.py` and run on two cores:

<details> <summary> Click to expand output</summary>

    ```console
    $ BODO_NUM_WORKERS=2 python load_data.py
    pandas dataframe:
                     A        B
    0              NaT        0
    1       2013-01-03        1
    2       2013-01-03        2
    3              NaT        3
    4       2013-01-03        4
    ...            ...      ...
    9999995 2015-09-29  9999995
    9999996 2015-09-29  9999996
    9999997 2015-09-29  9999997
    9999998 2015-09-29  9999998
    9999999 2015-09-29  9999999
    
    [10000000 rows x 2 columns]
    
    Bodo dataframe:
                     A        B
    0       1970-01-01        0
    1       2013-01-03        1
    2       2013-01-03        2
    3       2013-01-03        3
    4       2013-01-03        4
    ...            ...      ...
    4999995 2014-05-17  4999995
    4999996 2014-05-17  4999996
    4999997 2014-05-17  4999997
    4999998 2014-05-17  4999998
    4999999 2014-05-17  4999999
    
    [5000000 rows x 2 columns]
    
    5000000 2014-05-18  5000000
    5000001 2014-05-18  5000001
    5000002 2014-05-18  5000002
    5000003 2014-05-18  5000003
    5000004 2014-05-18  5000004
    ...            ...      ...
    9999995 2015-09-29  9999995
    9999996 2015-09-29  9999996
    9999997 2015-09-29  9999997
    9999998 2015-09-29  9999998
    9999999 2015-09-29  9999999
    
    [5000000 rows x 2 columns]
    ```

</details>   

The first dataframe is a regular Pandas dataframe and has all 10 million rows.
However, the second dataframe is a Bodo parallelized Pandas
dataframe which is split into two chunks with 5 million rows each. In this case,
Bodo parallelizes `read_parquet` automatically and loads different chunks of
data in different cores.

When parallelizable input is given to a function (e.g. Pandas DataFrames/Series,
numpy Arrays), Bodo will automatically distribute input for parallel
computation, freeing users from having to manually reason about transforming
data for parallel computation (see more on that below). For values returned by
JIT'ed functions, Bodo will avoid gathering all of the output back onto a single
process unless the full data is actually used outside of a parallel context.
This means that for large programs running across distributed clusters, one does
not need to worry about crashes due to running out of memory when retrieving a
large object from a JIT call.

### Parallel Computation

Bodo automatically divides computation and manages communication across
cores as this example demonstrates:

``` py
import pandas as pd
import bodo


@bodo.jit
def data_groupby():
    df = pd.read_parquet("pd_example.pq")
    df2 = df.groupby("A", as_index=False).sum()
    df2.to_parquet("bodo_output.pq")


if __name__ == "__main__":
    data_groupby()
```

Save this code as `data_groupby.py` and run from command line:

``` 
$ BODO_NUM_WORKERS=8 python data_groupby.py
```

This program uses `groupby` which requires rows with the same key to be
aggregated together. Therefore, Bodo *shuffles* the data automatically, and the
user doesn't need to worry about parallelism challenges
like communication.

<br>

![parallel processes](../img/python_parallel_process.svg#center)

Bodo JIT Requirements
---------------------

To take advantage of the Bodo JIT compiler and avoid errors,
make sure only compute and data-intensive
code is in JIT functions.
Other Python code for setup and configuration
should run in regular Python.
For example, consider this simple script:

```py
import os
import pandas as pd

data_path = os.environ["JOB_DATA_PATH"]

df = pd.read_parquet(data_path)
print(df.A.sum())
```

The Bodo version performs the computation in JIT functions,
but keeps the setup code (finding `data_path`) in regular Python:

```py
import os
import pandas as pd
import bodo

data_path = os.environ["JOB_DATA_PATH"]

@bodo.jit
def f(path):
    df = pd.read_parquet(path)
    print(df.A.sum())

f(data_path)
```

In addition, the Bodo version passes the file path `data_path` as an argument
to the JIT function `f`, allowing Bodo to find the input dataframe schema
which is necessary for type inference (more in [Scalable Data I/O][file_io]).


Bodo JIT supports specific APIs in Pandas currently, and other APIs
cannot be used inside JIT functions. For example:

``` py
import pandas as pd
import bodo


@bodo.jit
def df_unsupported():
    df = pd.DataFrame({"A": [1, 2, 3]})
    df2 = df.transpose()
    return df2


if __name__ == "__main__":
    df_unsupported()
```

Save this code as `df_unsupported.py` and run from command line:

``` 
$ python df_unsupported.py
# bodo.utils.typing.BodoError: Dataframe.transpose not supported yet
```

As the error indicates, Bodo doesn't currently support the `transpose`
call in JIT functions. In these cases, an alternative API should be used
or this portion of the code should be either be in regular Python or in Bodo's [Object Mode][objmode].
See [supported Pandas API][pandas] for the
complete list of supported Pandas operations.

### Type Stability

The main requirement of JIT compilation is being able to infer data
types for all variables and values. In Bodo, column names are part of
dataframe data types, so Bodo tries to infer column name related inputs
in all operations. For example, key names in `groupby` are used to
determine the output data type and need to be known to Bodo:

``` py
import pandas as pd
import bodo


@bodo.jit
def groupby_keys(extra_keys):
    df = pd.read_parquet("pd_example.pq")
    keys = [c for c in df.columns if c not in ["B", "C"]]
    if extra_keys:
        keys.append("B")
    df2 = df.groupby(keys).sum()
    print(df2)


if __name__ == "__main__":
    groupby_keys(False)
```

Save this code as `groupby_keys.py` and run from command line:

``` 
$ python groupby_keys.py
# bodo.utils.typing.BodoError: groupby(): argument 'by' requires a constant value but variable 'keys' is updated inplace using 'append'
```

In this case, the list of groupby keys is determined using the runtime
value of `extra_keys` in a way that Bodo is not able to infer it from
the program during compilation time. The alternative is to compute the
keys in a separate JIT function to make it easier for Bodo to infer:

``` py
import pandas as pd
import bodo


@bodo.jit
def get_keys(df_columns, extra_keys):
    keys = [c for c in df_columns if c not in ["B", "C"]]
    if extra_keys:
        keys.append("B")
    return keys


@bodo.jit
def groupby_keys(extra_keys):
    df = pd.read_parquet("pd_example.pq")
    keys = get_keys(df.columns, extra_keys)
    df2 = df.groupby(keys).sum()
    print(df2)


if __name__ == "__main__":
    keys = get_keys(["A"], False)
    groupby_keys(False)
```

This program works since `get_keys` can be evaluated in compile time. It
only uses `df.columns` and `extra_keys` values that can be constant at
compile time, and does not use non-deterministic features like I/O.

### Python Features

Bodo uses [Numba](http://numba.pydata.org){target="blank"} for compiling regular Python
features and some of Numba's requirements apply to Bodo as well. For
example, values in data structures like lists should have the same data
type. This example fails since list values are either integers or
strings:

``` py
import bodo


@bodo.jit
def create_list():
    out = []
    out.append(0)
    out.append("A")
    out.append(1)
    out.append("B")
    return out


if __name__ == "__main__":
    create_list()
```

Using tuples can often solve these problems since tuples can hold values
of different types:

``` py
import bodo


@bodo.jit
def create_list():
    out = []
    out.append((0, "A"))
    out.append((1, "B"))
    return out


if __name__ == "__main__":
    create_list()
```

Please refer to the [Unsupported Python Programs](../bodo_parallelism/not_supported.md#notsupported)
documentation for more details.

Using Bodo in Jupyter Notebooks {#jupyter}
-------------------------------

See [Interactive Bodo Cluster Setup using IPyParallel][ipyparallelsetup] for more
information.

Using Bodo in Single Program Multiple Data (SPMD) mode
-------------------------------

For some advanced users, there may be programs that are easier to write in SPMD
style where the entire program is run on all cores. Under the hood, Bodo
leverages
[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface){target="blank"}
to manage parallel execution. This means that Bodo can be launched as an MPI
process and be informed to use existing processes instead of launching a new
group of workers.

Let's revisit the same example from before, but with SPMD mode:

``` py
import pandas as pd
import bodo


def load_data_pandas():
    df = pd.read_parquet("pd_example.pq")
    print("pandas dataframe: ", df)


# By setting spawn=False, we inform Bodo to use the existing set of MPI
# processes
@bodo.jit(spawn=False)
def load_data_bodo():
    df = pd.read_parquet("pd_example.pq")
    print("Bodo dataframe: ", df)


if __name__ == "__main__":
    load_data_pandas()
    load_data_bodo()
```

Save this code in `load_data.py` and run on two cores (output prints of
the cores are mixed):

<details> <summary> Click to expand output</summary>

    ```console
    $ mpiexec -n 2 python load_data.py
    pandas dataframe:
                     A        B
    0              NaT        0
    1       2013-01-03        1
    2       2013-01-03        2
    3              NaT        3
    4       2013-01-03        4
    ...            ...      ...
    9999995 2015-09-29  9999995
    9999996 2015-09-29  9999996
    9999997 2015-09-29  9999997
    9999998 2015-09-29  9999998
    9999999 2015-09-29  9999999
    
    [10000000 rows x 2 columns]
    
    pandas dataframe:
                     A        B
    0              NaT        0
    1       2013-01-03        1
    2       2013-01-03        2
    3              NaT        3
    4       2013-01-03        4
    ...            ...      ...
    9999995 2015-09-29  9999995
    9999996 2015-09-29  9999996
    9999997 2015-09-29  9999997
    9999998 2015-09-29  9999998
    9999999 2015-09-29  9999999
    
    [10000000 rows x 2 columns]
    
    Bodo dataframe:
                     A        B
    0       1970-01-01        0
    1       2013-01-03        1
    2       2013-01-03        2
    3       2013-01-03        3
    4       2013-01-03        4
    ...            ...      ...
    4999995 2014-05-17  4999995
    4999996 2014-05-17  4999996
    4999997 2014-05-17  4999997
    4999998 2014-05-17  4999998
    4999999 2014-05-17  4999999
    
    [5000000 rows x 2 columns]
    
    pandas dataframe:
                     A        B
    5000000 2014-05-18  5000000
    5000001 2014-05-18  5000001
    5000002 2014-05-18  5000002
    5000003 2014-05-18  5000003
    5000004 2014-05-18  5000004
    ...            ...      ...
    9999995 2015-09-29  9999995
    9999996 2015-09-29  9999996
    9999997 2015-09-29  9999997
    9999998 2015-09-29  9999998
    9999999 2015-09-29  9999999
    
    [5000000 rows x 2 columns]
    ```

</details>   

`mpiexec` launches identical Python processes but Bodo divides the
data and computation in JIT functions to exploit parallelism.

Unlike default mode, in SPMD mode, the regular pandas dataframe with all 10 million
rows initially exists on both cores. Because each core is running the full
program, we see the pandas dataframe printed twice. 

The last two dataframes printed are Bodo parallelized Pandas dataframes, with 5
million rows each. In this case, Bodo parallelizes `read_parquet` automatically
and loads different chunks of data in different cores. Therefore, the non-JIT
parts of the Python program are replicated across cores whereas Bodo JIT
functions are parallelized.

One of the challenges of using SPMD mode is that third-party libraries are
usually not aware of the MPI runtime, and may not expect to be run on every
core. A common workaround is to use `bodo.get_rank()` to get a unique ID per
core (integers from `0` to `num processes`), and conditionally execute code only
when the rank is `0`.

The advantage of SPMD mode is that operations outside of JIT functions that do
not need access to the full table are inherently parallelized. Since each core
only has a chunk of the data, when these chunks are returned to regular python
from a JIT function, subsequent operations will happen on all ranks to a subset
of the full dataframe. However, this can be error-prone, as operations that
require access to all rows will need explicit communication. Consider the
following:

``` py
import pandas as pd
import bodo


@bodo.jit(spawn=False)
def load_data_bodo():
    df = pd.read_parquet("pd_example.pq")
    return df["B"]


if __name__ == "__main__":
    col_B = load_data_bodo()
    print("sum", col_B.sum())
```

Saving the above as `example.py` and running with 2 cores:

```console
$ mpiexec -n 2 python load_data.py
sum 12499997500000
sum 37499997500000
```

We see that each core prints out the sum of their local data. However, in most
cases we likely want the sum of the entire table. The ideal way to do this is to
move the `.sum()` call into the JIT function so that Bodo can efficiently
parallelize the `sum` and distribute the results. However, to do this outside of
the JIT function, one would need to handle the communication explicitly:

``` py
import pandas as pd
import bodo
from bodo.mpi4py import MPI


@bodo.jit(spawn=False)
def load_data_bodo():
    df = pd.read_parquet("pd_example.pq")
    return df["B"]


if __name__ == "__main__":
    col_B = load_data_bodo()
    local_sum = col_B.sum()

    comm = MPI.COMM_WORLD
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    print("sum", global_sum)
```

In some cases, SPMD mode can also have lower overhead by eliminating the need to
communicate between the spawned processes and the main application. However,
this tends to not be a significant source of overhead unless the application is
very short-lived.

To summarize, SPMD mode allows non-JIT code to operate on chunks of data, but
makes it harder to write correct programs (e.g. there are risks around code that
is unaware that it's running in a distributed environment). In most cases, it is
preferable to use the default mode (and use object mode within JIT functions to
call non-JIT libraries on chunks of data).
