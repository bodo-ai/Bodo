Bodo Parallelism Basics
-------------------------

In this section, we will discuss Bodo's JIT compilation workflow and the parallelism model and APIs provided by Bodo.

.. _jit:

JIT (Just-in-time) Compilation Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~

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

.. figure:: ../img/barrier.svg
   :align: center
   :alt: Process synchronization with Barrier

   Process synchronization with Barrier

.. danger::

    The examples above show that it is possible to have each process follow a different control flow, but all processes must always call the same Bodo functions in the same order.
