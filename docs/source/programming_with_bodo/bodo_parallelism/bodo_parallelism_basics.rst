.. _basics:

Bodo Parallelism Basics
========================

In this section, we will discuss Bodo's JIT compilation workflow and the parallelism model and APIs provided by Bodo.

.. _jit:

JIT (Just-in-time) Compilation Workflow
----------------------------------------

Bodo provides a just-in-time (JIT) compilation workflow using the
``@bodo.jit`` decorator, which replaces a Python function with a
so-called ``Dispatcher`` object. Bodo compiles the function the first
time a Dispatcher object is called and reuses the compiled version
afterwards. The function is recompiled only if the same function is
called with different argument types (not often in practice).

.. code:: ipython3

    import numpy as np
    import pandas as pd
    import bodo

    @bodo.jit
    def f(n, a):
        df = pd.DataFrame({"A": np.arange(n) + a})
        return df.head(3)

    print(f)
    print(f(8, 1))  # compiles for (int, int) input types
    print(f(8, 2))  # same input types, no need to compile
    print(f(8, 2.2))  # compiles for (int, float) input types


.. parsed-literal::

    CPUDispatcher(<function f at 0x7fa35e533c10>)
       A
    0  1
    1  2
    2  3
       A
    0  2
    1  3
    2  4
         A
    0  2.2
    1  3.2
    2  4.2


All of this is completely transparent to the caller, and does not affect
any Python code calling the function.

.. note::


   In many cases, the binary that Bodo generates when compiling a function can be saved to disk to be reused across program executions. See :ref:`Bodo Caching <caching>` for more information.



Parallel Execution Model
-------------------------

As we saw in the “Getting Started” tutorial, Bodo transforms functions
for parallel execution. However, the dispatcher does not launch
processes or threads on the fly. Instead, the Python application
(including non-Bodo code) is intended to be executed under an MPI Single
Program Multiple Data (`SPMD <https://en.wikipedia.org/wiki/SPMD>`__)
paradigm, where MPI processes are launched in the beginning and all run
the same code.

For example, we can save an example code in a file and use *mpiexec* to
launch 4 processes:

.. code:: ipython3

    import numpy as np
    import pandas as pd
    import bodo

    @bodo.jit(distributed=["df"])
    def f(n, a):
        df = pd.DataFrame({"A": np.arange(n) + a})
        return df

    print(f(8, 1))

.. code:: ipython3

    %save -f test_bodo.py 2 # cell number of previous cell

.. code:: ipython3

    !mpiexec -n 4 python test_bodo.py


.. parsed-literal::

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


In this example, ``mpiexec`` launches 4 Python processes, each of which
executes the same ``test_bodo.py`` file.

.. warning::



   - Python codes outside of Bodo functions execute sequentially on every process.
   - Bodo functions run in parallel assuming that Bodo is able to parallelize them. Otherwise, they also run sequentially on every process. Bodo warns if it does not find parallelism (more details later).



Note how the prints, which are regular Python code executed outside of
Bodo, run for each process.

On Jupyter notebook, parallel execution happens in very much the same
way. We start a set of MPI engines through ``ipyparallel`` and activate
a client (*NOTE: if you are using the Bodo Platform, this is already
done automatically*):

.. code:: ipython3

    import ipyparallel as ipp
    c = ipp.Client(profile="mpi")
    view = c[:]
    view.activate()
    view.block = True
    import os
    view["cwd"] = os.getcwd()
    %px cd $cwd

After this initialization, any code that we run in the notebook with
``%%px`` is sent for execution on all MPI engines.

.. code:: ipython3


    import numpy as np
    import pandas as pd
    import bodo

    @bodo.jit(distributed=["df"])
    def f(n):
        df = pd.DataFrame({"A": np.arange(n)})
        return df

    print(f(8))


.. parsed-literal::

    [stdout:0]
       A
    0  0
    1  1
    [stdout:1]
       A
    2  2
    3  3
    [stdout:2]
       A
    4  4
    5  5
    [stdout:3]
       A
    6  6
    7  7


Parallel APIs
--------------

Bodo provides a limited number of parallel APIs to support advanced
cases that may need them. The example below demonstrates getting the
process number from Bodo (called ``rank`` in MPI terminology) and the
total number of processes.

.. code:: ipython3


    # some work only on rank 0
    if bodo.get_rank() == 0:
        print("rank 0 done")

    # some work on every process
    print("rank", bodo.get_rank(), "here")
    print("total ranks:", bodo.get_size())


.. parsed-literal::

    [stdout:0]
    rank 0 done
    rank 0 here
    total ranks: 4
    [stdout:1]
    rank 1 here
    total ranks: 4
    [stdout:2]
    rank 2 here
    total ranks: 4
    [stdout:3]
    rank 3 here
    total ranks: 4


A common pattern is using barriers to make sure all processes see
side-effects at the same time. For example, a process can delete files
from storage while others wait before writing to file:

.. code:: ipython3

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

The following figure illustrates what happens when processes call
``bodo.barrier()``. When barrier is called, a process pauses and waits
until all other processes have reached the barrier:

.. figure:: ../../img/barrier.svg
   :align: center
   :alt: Process synchronization with Barrier

   Process synchronization with Barrier

.. danger::

    The examples above show that it is possible to have each process follow a different control flow, but all processes must always call the same Bodo functions in the same order.


Data Distribution
-----------------

Bodo parallelizes computation by dividing data into separate chunks
across processes. However, some data handled by a Bodo function may not
be divided into chunks. There are are two main data distribution
schemes:

-  Replicated (*REP*): the data associated with the variable is the same
   on every process.
-  One-dimensional (*1D*): the data is divided into chunks, split along
   one dimension (rows of a dataframe or first dimension of an array).

Bodo determines distribution of variables automatically, using the
nature of the computation that produces them. Let’s see an example:

.. code:: ipython3


    @bodo.jit
    def mean_power_speed():
        df = pd.read_parquet("data/cycling_dataset.pq")
        m = df[["power", "speed"]].mean()
        return m

    res = mean_power_speed()
    print(res)


.. parsed-literal::

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


In this example, ``df`` is parallelized (each process reads a different
chunk) but ``m`` is replicated, even though it is a Series.
Semantically, it makes sense for the output of ``mean`` operation to be
replicated on all processors, since it is a reduction and produces
“small” data.

Distributed Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~

The distributions found by Bodo can be printed either by setting the
environment variable ``BODO_DISTRIBUTED_DIAGNOSTICS=1`` or calling
``distributed_diagnostics()`` on the compiled function. Let’s examine
the previous example’s distributions:

.. code:: ipython3

    mean_power_speed.distributed_diagnostics()


.. parsed-literal::

    [stdout:0]
    Distributed diagnostics for function mean_power_speed, <ipython-input-29-0669fd25a56c> (1)

    Data distributions:
       power.10770                1D_Block
       speed.10771                1D_Block
       $A.10860.11199             1D_Block
       $A.10923.11209             1D_Block
       $data.10832.11220          REP
       $12call_method.5.11183     REP
       $66call_method.31.10850    REP
       $m.11222                   REP
       $30return_value.12         REP

    Parfor distributions:
       31                   1D_Block
       30                   1D_Block

    Distributed listing for function mean_power_speed, <ipython-input-29-0669fd25a56c> (1)
    -------------------------------------------------------| parfor_id/variable: distribution
    @bodo.jit                                              |
    def mean_power_speed():                                |
        df = pd.read_parquet("data/cycling_dataset.pq")----| power.10770: 1D_Block, speed.10771: 1D_Block
        m = df[["power", "speed"]].mean()------------------| $A.10860.11199: 1D_Block, $A.10923.11209: 1D_Block
        return m-------------------------------------------| $30return_value.12: REP

    Distributed analysis replicated return variable $30return_value.12. Set distributed flag for the original variable if distributed partitions should be returned.



Variables are renamed due to optimization. The output shows that
``power`` and ``speed`` columns of ``df`` are distributed (``1D_Block``)
but ``m`` is replicated (``REP``). This is because ``df`` is output of
``read_parquet`` and input of ``mean``, both of which can be distributed
by Bodo. ``m`` is output of ``mean``, which is always replicated
(available on every process).


Scattering Data
~~~~~~~~~~~~~~~

One can distribute data manually by *scattering* data from one process
to all processes with scatterv. Currently, bodo.scatterv only supports scattering from rank 0.
When used outside of JIT code, it is recommended that
the argument is None for all ranks except rank 0. For example:

.. code:: ipython3


    @bodo.jit(distributed=["df"])
    def mean_power(df):
        x = df.power.mean()
        return x

    df = None
    # only rank 0 reads the data
    if bodo.get_rank() == 0:
        df = pd.read_parquet("data/cycling_dataset.pq")

    df = bodo.scatterv(df)
    res = mean_power(df)
    print(res)


.. parsed-literal::

    [stdout:0] 102.07842132239877
    [stdout:1] 102.07842132239877
    [stdout:2] 102.07842132239877
    [stdout:3] 102.07842132239877

This is not a strict requirement. However, since this might be bad practice in certain situations, Bodo will throw a warning if the data is not None on other ranks.


.. code:: ipython3


    df = pd.read_parquet("data/cycling_dataset.pq")
    df = bodo.scatterv(df)
    res = mean_power(df)
    print(res)


.. parsed-literal::
    BodoWarning: bodo.scatterv(): A non-None value for 'data' was found on a rank other than the root. This data won't be sent to any other ranks and will be overwritten with data from rank 0.
    
    [stdout:0] 102.07842132239877
    [stdout:1] 102.07842132239877
    [stdout:2] 102.07842132239877
    [stdout:3] 102.07842132239877

When using scatterv inside of JIT code, the argument must have the same type on each rank due to Bodo's typing constraints.
All inputs except for rank 0 are ignored.


.. code:: ipython3


    @bodo.jit()
    def impl():
        if bodo.get_rank() == 0:
            df = pd.DataFrame({"A": [1,2,3,4,5,6,7,8]})
        else:
            df = pd.DataFrame({"A": [-1]*8})
        return bodo.scatterv(df)
    print(impl())


.. parsed-literal::

    [stdout:6]
          A
    6     7
    [stdout:0]
          A
    0     1
    [stdout:1]
          A
    1     2
    [stdout:4]
          A
    4     5
    [stdout:7]
          A
    7     8
    [stdout:3]
          A
    3     4
    [stdout:2]
          A
    2     3
    [stdout:5]
          A
    5     6


Gathering Data
~~~~~~~~~~~~~~

One can *gather* distributed data into a single process manually. The rank into which the data is gathered can be
changed by using the root keyword argument (defaults to rank 0). For
example:

.. code:: ipython3


    @bodo.jit
    def mean_power():
        df = pd.read_parquet("data/cycling_dataset.pq")
        return bodo.gatherv(df, root=1)

    df = mean_power()
    print(df)


.. parsed-literal::

    [stdout:1]
          Unnamed: 0    altitude  cadence  ...  power  speed                time
    0              0  185.800003       51  ...     45  3.459 2016-10-20 22:01:26
    1              1  185.800003       68  ...      0  3.710 2016-10-20 22:01:27
    2              2  186.399994       38  ...     42  3.874 2016-10-20 22:01:28
    3              3  186.800003       38  ...      5  4.135 2016-10-20 22:01:29
    4              4  186.600006       38  ...      1  4.250 2016-10-20 22:01:30
    ...          ...         ...      ...  ...    ...    ...                 ...
    3897        1127  178.199997        0  ...      0  3.497 2016-10-20 23:14:31
    3898        1128  178.199997        0  ...      0  3.289 2016-10-20 23:14:32
    3899        1129  178.199997        0  ...      0  2.969 2016-10-20 23:14:33
    3900        1130  178.399994        0  ...      0  2.969 2016-10-20 23:14:34
    3901        1131  178.399994        0  ...      0  2.853 2016-10-20 23:14:35

    [3902 rows x 10 columns]
    [stdout:0]
    Empty DataFrame
    Columns: [Unnamed: 0, altitude, cadence, distance, hr, latitude, longitude, power, speed, time]
    Index: []

    [0 rows x 10 columns]
    [stdout:2]
    Empty DataFrame
    Columns: [Unnamed: 0, altitude, cadence, distance, hr, latitude, longitude, power, speed, time]
    Index: []

    [0 rows x 10 columns]
    [stdout:3]
    Empty DataFrame
    Columns: [Unnamed: 0, altitude, cadence, distance, hr, latitude, longitude, power, speed, time]
    Index: []

    [0 rows x 10 columns]


Alternatively, distributed data can be gathered and sent to all
processes, effectively replicating the data:

.. code:: ipython3


    @bodo.jit
    def mean_power():
        df = pd.read_parquet("data/cycling_dataset.pq")
        return bodo.allgatherv(df)

    df = mean_power()
    print(df)


.. parsed-literal::

    [stdout:0]
          Unnamed: 0    altitude  cadence  ...  power  speed                time
    0              0  185.800003       51  ...     45  3.459 2016-10-20 22:01:26
    1              1  185.800003       68  ...      0  3.710 2016-10-20 22:01:27
    2              2  186.399994       38  ...     42  3.874 2016-10-20 22:01:28
    3              3  186.800003       38  ...      5  4.135 2016-10-20 22:01:29
    4              4  186.600006       38  ...      1  4.250 2016-10-20 22:01:30
    ...          ...         ...      ...  ...    ...    ...                 ...
    3897        1127  178.199997        0  ...      0  3.497 2016-10-20 23:14:31
    3898        1128  178.199997        0  ...      0  3.289 2016-10-20 23:14:32
    3899        1129  178.199997        0  ...      0  2.969 2016-10-20 23:14:33
    3900        1130  178.399994        0  ...      0  2.969 2016-10-20 23:14:34
    3901        1131  178.399994        0  ...      0  2.853 2016-10-20 23:14:35

    [3902 rows x 10 columns]
    [stdout:1]
          Unnamed: 0    altitude  cadence  ...  power  speed                time
    0              0  185.800003       51  ...     45  3.459 2016-10-20 22:01:26
    1              1  185.800003       68  ...      0  3.710 2016-10-20 22:01:27
    2              2  186.399994       38  ...     42  3.874 2016-10-20 22:01:28
    3              3  186.800003       38  ...      5  4.135 2016-10-20 22:01:29
    4              4  186.600006       38  ...      1  4.250 2016-10-20 22:01:30
    ...          ...         ...      ...  ...    ...    ...                 ...
    3897        1127  178.199997        0  ...      0  3.497 2016-10-20 23:14:31
    3898        1128  178.199997        0  ...      0  3.289 2016-10-20 23:14:32
    3899        1129  178.199997        0  ...      0  2.969 2016-10-20 23:14:33
    3900        1130  178.399994        0  ...      0  2.969 2016-10-20 23:14:34
    3901        1131  178.399994        0  ...      0  2.853 2016-10-20 23:14:35

    [3902 rows x 10 columns]
    [stdout:2]
          Unnamed: 0    altitude  cadence  ...  power  speed                time
    0              0  185.800003       51  ...     45  3.459 2016-10-20 22:01:26
    1              1  185.800003       68  ...      0  3.710 2016-10-20 22:01:27
    2              2  186.399994       38  ...     42  3.874 2016-10-20 22:01:28
    3              3  186.800003       38  ...      5  4.135 2016-10-20 22:01:29
    4              4  186.600006       38  ...      1  4.250 2016-10-20 22:01:30
    ...          ...         ...      ...  ...    ...    ...                 ...
    3897        1127  178.199997        0  ...      0  3.497 2016-10-20 23:14:31
    3898        1128  178.199997        0  ...      0  3.289 2016-10-20 23:14:32
    3899        1129  178.199997        0  ...      0  2.969 2016-10-20 23:14:33
    3900        1130  178.399994        0  ...      0  2.969 2016-10-20 23:14:34
    3901        1131  178.399994        0  ...      0  2.853 2016-10-20 23:14:35

    [3902 rows x 10 columns]
    [stdout:3]
          Unnamed: 0    altitude  cadence  ...  power  speed                time
    0              0  185.800003       51  ...     45  3.459 2016-10-20 22:01:26
    1              1  185.800003       68  ...      0  3.710 2016-10-20 22:01:27
    2              2  186.399994       38  ...     42  3.874 2016-10-20 22:01:28
    3              3  186.800003       38  ...      5  4.135 2016-10-20 22:01:29
    4              4  186.600006       38  ...      1  4.250 2016-10-20 22:01:30
    ...          ...         ...      ...  ...    ...    ...                 ...
    3897        1127  178.199997        0  ...      0  3.497 2016-10-20 23:14:31
    3898        1128  178.199997        0  ...      0  3.289 2016-10-20 23:14:32
    3899        1129  178.199997        0  ...      0  2.969 2016-10-20 23:14:33
    3900        1130  178.399994        0  ...      0  2.969 2016-10-20 23:14:34
    3901        1131  178.399994        0  ...      0  2.853 2016-10-20 23:14:35

    [3902 rows x 10 columns]

You can also get identical behavior by using gatherv and setting the keyword aregument allgatherv to True.

scatterv, gatherv, and and allgatherv work with all distributable data types. This includes:
 * All supported numpy array types
 * All supported pandas array types (with the exception of Interval Arrays)
 * All supported pandas Series types
 * All supported DataFrame types
 * All supported Index types (with the exception of Interval Index)
 * Tuples of the above types
