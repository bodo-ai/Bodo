# bodo.barrier

`bodo.barrier()`

Synchronize all processes. Block process from proceeding until all processes reach this point.

### Example Usage

A typical example is to make sure all processes see side effects simultaneously.
For example, a process can delete files from storage while
others wait before writing to file.
The following example uses [SPMD launch mode][spmd]:

```py
import shutil, os
import numpy as np

# remove file if exists
if bodo.get_rank() == 0:
    if os.path.exists("data/data.pq"):
        shutil.rmtree("data/data.pq")

# make sure all processes are synchronized
# (e.g. all processes need to see effect of rank 0's work)
bodo.barrier()


@bodo.jit
def f(n):
    df = pd.DataFrame({"A": np.arange(n)})
    df.to_parquet("data/data.pq")


f(10)
```

The following figure illustrates what happens when processes call
`bodo.barrier()`. When barrier is called, a process pauses and waits
until all other processes have reached the barrier:

![Process synchronization with Barrier](../../img/barrier.svg#center)

!!! danger
    The example above shows that it is possible to have each process
    follow a different control flow, but all processes must always call
    the same Bodo functions in the same order.

