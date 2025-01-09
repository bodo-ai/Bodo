# Caching

In many situations, Bodo can save the binary resulting from the
compilation of a function to disk, to be reused in future runs. This
avoids the need to recompile functions the next time that you run your
application.

Recompiling a function is only necessary when it is called with new
input types, and the same applies to caching. In other words, an
application can be run multiple times and process different data without
having to recompile any code if the data types remain the same (which is
the most common situation).

!!! warning

    Caching works in most (but not all) situations, and is disabled by
    default. See caching limitations below for more information.


## Caching Example

To cache a function, we only need to add the option `cache=True` to the
JIT decorator:

```py
import time
import pandas as pd
import bodo


@bodo.jit(cache=True)
def mean_power_speed():
    df = pd.read_parquet("data/cycling_dataset.pq")
    return df[["power", "speed"]].mean()


t0 = time.time()
result = mean_power_speed()
print(result)
print("Total execution time:", round(time.time() - t0, 3), "secs")
```

The first time that the above code runs, Bodo compiles the function and
caches it to disk. The code times the whole function call, which
includes compilation time the first time the function is run:

```console
power    102.078421
speed      5.656851
dtype: float64
Total execution time: 4.614 secs
```
In subsequent runs, it will recover the function from cache and as a
result, the execution time will be much faster:

```console
power    102.078421
speed      5.656851
dtype: float64
Total execution time: 0.518 secs
```

!!! note
    `data/cycling_dataset.pq` is located in the Bodo tutorial
    [repo](https://github.com/bodo-ai/Bodo-tutorial).


## Cache Location and Portability

In most cases, the cache is saved in the `__pycache__` directory inside
the directory where the source files are located. The variable
`NUMBA_DEBUG_CACHE` can be set to `1` in order to see where exactly the
cache is and whether it is being written to or read from.

On Jupyter notebooks, the cache directory is called `numba_cache` and is
located in `IPython.paths.get_ipython_cache_dir()`. See
[here](http://numba.pydata.org/numba-doc/latest/reference/envvars.html?#envvar-NUMBA_CACHE_DIR)
for more information on these and other alternate cache locations. For
example, when running in a notebook:

```py
import os
import IPython


cache_dir = IPython.paths.get_ipython_cache_dir() + "/numba_cache"
print("Cache files:")
os.listdir(cache_dir)
```

```console
Cache files:
['ipython-input-bce41f829e09.mean_power_speed-4444615264.py38.nbi',
'ipython-input-bce41f829e09.mean_power_speed-4444615264.py38.1.nbc']
```

Cached objects work across systems with the same CPU model and CPU
features. Therefore, it is safe to share and reuse the contents in the
cache directory on a different machine. See
[here](http://numba.pydata.org/numba-doc/latest/developer/caching.html#cache-sharing){target="blank"}
for more information.

## Cache Invalidation

The cache is invalidated automatically when the corresponding source
code is modified. One way to observe this behavior is to modify the
above example after it has been cached a first time, by changing the
name of the variable `df`. The next time that we run the code, Bodo will
determine that the source code has been modified, invalidate the cache
and recompile the function.

!!! warning
    It is sometimes necessary to clear the cache manually (see caching
    limitations below). To clear the cache, the cache files can simply be
    removed.


## Tips for Reusing the Cache

As explained above, caching is invalidated for a function any time any
of the source code in the file changes. If we define a function and call
it in the same file, and modify the arguments passed to the function,
the cache will be invalidated.

### Caching File IO

For example: a typical use case is calling an IO function with a
different file name.

``` py
@bodo.jit(cache=True)
def io_call(file_name):
    ...
io_call("mydata.parquet")
```

The above function would need to be recompiled if the argument to
`io_call` changes from `mydata.parquet`. By separating into separate
files the function call from the function definition, the function
definition does not need to be recompiled for each function call with
new arguments. The cached IO function will work for a change in file
name so long as the file schema is the same. For example, the below code
snippet

``` py
import IO_function from IO_functions
IO_function(file_name)
```

would not need to recompile `IO_function` each time `file_name` is
modified since `IO_function` is isolated from that code change.

### Caching Notebook Cells

For IPython notebooks the function to be cached should be in a separate
cell from the function call.

``` py
@bodo.jit(cache=True)
def io_call(file_name):
    ...
```

``` py
io_call(file_name)
io_call(another_file_name)
...
```

If a cell with a cached function is modified, then its cache is
invalidated and the function must be compiled again.

## Current Caching Limitations

-   Changes in compiled functions are not seen across files. For
    example, if we have a cached Bodo function that calls a cached Bodo
    function in a different file, and modify the latter, Bodo will not
    update its cache (and therefore run with the old version of the
    function).
-   Global variables are treated as compile-time constants. When a
    function is compiled, the value of any globals that the function
    uses are embedded in the binary at compilation time and remain
    constant. If the value of the global changes in the source code
    after compilation, the compiled object (and cache) will not rebind
    to the new value.

## Troubleshooting

During execution, Bodo will print information on caching if the
environment variable `NUMBA_DEBUG_CACHE` is set to `1`. For example, on
first run it will show if the cache is being saved to and where, and on
subsequent runs it will show if the compiler is successfully loading
from cache.

If the compiler reports that it is not able to cache a function, or load
a function from cache, please report the issue
[on our respository](https://github.com/bodo-ai/Bodo).
