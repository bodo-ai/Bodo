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
All of this is completely transparent to the caller, and does not affect
any Python code calling the function.

.. code::

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

.. note::

   In many cases, the binary that Bodo generates when compiling a function can be saved to disk and reused across program executions.
   See :ref:`caching` for more information.

Parallel Execution Model
-------------------------

As we saw in the “Getting Started” tutorial, Bodo transforms functions
for parallel execution. Bodo uses Message Passing Interface (`MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_) 
that follows Single Program Multiple Data (`SPMD <https://en.wikipedia.org/wiki/SPMD>`_) paradigm.
In this model, the dispatcher does not launch processes or threads on the fly.
Instead, all processes are launched at the beginning and run the same file using ``mpiexec`` command.

Bodo parallelizes functions that have ``bodo.jit`` decorator by distributing the data across the processes.
Each rank runs the same code on a chunk of the data and Bodo automatically communicates the data between the ranks (as needed).

For example, save the following code in a``test_bodo.py`` and use ``mpiexec`` to
launch 4 processes as follows:

.. code::

    import numpy as np
    import pandas as pd
    import bodo

    @bodo.jit
    def f(n, a):
        df = pd.DataFrame({"A": np.arange(n) + a})
        return df

    print(f(8, 1))

.. code::

    mpiexec -n 4 python test_bodo.py

Output:

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
Since the function ``f`` is decorated with ``bodo.jit`` and Bodo is able to parallelize it, each process generates a chunk of the data in ``np.arange`` 

Note how the prints, which are regular Python code executed outside of Bodo, run for each process.

  .. warning::
    - Python codes outside of Bodo functions execute sequentially on every process.
    - Bodo functions run in parallel assuming that Bodo is able to parallelize them. 
      Otherwise, Bodo prints the following warning and runs sequentially on every process. 

      ``BodoWarning: No parallelism found for function``

On Jupyter notebook, parallel execution happens in very much the same
way. We start a set of MPI engines through ``ipyparallel`` and activate
a client. See :ref:`jupyter` for more information and examples.


Parallel APIs
--------------

Even though Bodo handles all the parallel communications, Bodo provides a limited number of parallel APIs to support 
cases where users may need to explicitly do some communications. 

    - ``get_rank()``: Get the process number from Bodo (called ``rank`` in MPI terminology).

    - ``get_size()``: Get the total number of processes.

    Example: Save following code in ``get_rank_size.py`` file and run with ``mpiexec``.

    .. code:: 


        import bodo
        # some work only on rank 0
        if bodo.get_rank() == 0:
            print("rank 0 done")

        # some work on every process
        print("rank", bodo.get_rank(), "here")
        print("total ranks:", bodo.get_size())

    .. code::

        mpiexec -n 4 python get_rank_size.py

    .. parsed-literal::

        rank 0 done
        rank 0 here
        total ranks: 4
        rank 1 here
        total ranks: 4
        rank 2 here
        total ranks: 4
        rank 3 here
        total ranks: 4


    - ``barrier()`` : synchronize all processes. Block process from proceeding until all processes reach this point.

    A common example is to make sure all processes see
    side-effects at the same time. For example, a process can delete files
    from storage while others wait before writing to file:

    .. code:: 

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
        The examples above show that it is possible to have each process follow a different control flow, 
        but all processes must always call the same Bodo functions in the same order.

  - ``scatterv(data, warn_if_dist=True)`` : Distribute data manually by *scattering* data from one process to all processes.

    Arguments:
        ``data``: data to distribute.

        ``warn_if_dist``: flag to print a BodoWarning if ``data`` is already distributed.

    .. note::
        Currently, ``bodo.scatterv`` only supports scattering from rank 0.

  When used outside of JIT code, it is recommended that
  the argument is ``None`` for all ranks except rank 0. For example:

    .. code:: 

        import bodo
        import pandas as pd


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

    Save code in ``test_scatterv.py`` file and run with **mpiexec**.

    .. code::

        mpiexec -n 4 python test_scatterv.py

    .. parsed-literal::

        [stdout:0] 102.07842132239877
        [stdout:1] 102.07842132239877
        [stdout:2] 102.07842132239877
        [stdout:3] 102.07842132239877

    .. note::

        ``data/cycling_dataset.pq`` is located in the Bodo tutorial `repo
        <https://github.com/Bodo-inc/Bodo-tutorial>`_.

  This is not a strict requirement. However, since this might be bad practice in certain situations, Bodo will throw a warning if the data is not None on other ranks.


    .. code:: 

        import bodo
        import pandas as pd

        df = pd.read_parquet("data/cycling_dataset.pq")
        df = bodo.scatterv(df)
        res = mean_power(df)
        print(res)

    Save code in ``test_scatterv.py`` file and run with **mpiexec**.

    .. code::

        mpiexec -n 4 python test_scatterv.py


    .. parsed-literal::
        BodoWarning: bodo.scatterv(): A non-None value for 'data' was found on a rank other than the root. This data won't be sent to any other ranks and will be overwritten with data from rank 0.
    
        [stdout:0] 102.07842132239877
        [stdout:1] 102.07842132239877
        [stdout:2] 102.07842132239877
        [stdout:3] 102.07842132239877

    When using ``scatterv`` inside of JIT code, the argument must have the same type on each rank due to Bodo's typing constraints.
    All inputs except for rank 0 are ignored.


    .. code:: 

            import bodo
            import pandas as pd

            @bodo.jit()
            def impl():
                if bodo.get_rank() == 0:
                    df = pd.DataFrame({"A": [1,2,3,4,5,6,7,8]})
                else:
                    df = pd.DataFrame({"A": [-1]*8})
                return bodo.scatterv(df)
            print(impl())

    Save code in ``test_scatterv.py`` file and run with **mpiexec**.

    .. code::

            mpiexec -n 8 python test_scatterv.py


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

    - ``gatherv(data, allgather=False, warn_if_rep=True, root=0)``: Collect distributed data manually by *gathering* them into a single rank. 

        Arguments:

            ``data``: data to gather.

            ``root``: specify rank to collect the data. Default: rank `0`.

            ``warn_if_rep``: prints a BodoWarning if data to gather is replicated. 

            ``allgather``: send gathered data to all ranks. Default: `False`. Same behavior as ``bodo.allgatherv``.

        .. code:: 

            import bodo
            import pandas as pd

            @bodo.jit
            def mean_power():
                df = pd.read_parquet("data/cycling_dataset.pq")
                return bodo.gatherv(df, root=1)

            df = mean_power()
            print(df)

        Save code in ``test_gatherv.py`` file and run with **mpiexec**.

        .. code::

                mpiexec -n 4 python test_gatherv.py

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



    - ``allgatherv(data, warn_if_rep=True)`` : gather data from all ranks and send to all, effectively replicating the data:

        Arguments:

            ``data``: data to gather.

            ``warn_if_rep``: prints a BodoWarning if data to gather is replicated. 

        .. code::

            import bodo
            import pandas as pd

            @bodo.jit
            def mean_power():
                df = pd.read_parquet("data/cycling_dataset.pq")
                return bodo.allgatherv(df)

            df = mean_power()
            print(df)

        Save code in ``test_allgatherv.py`` file and run with **mpiexec**.

        .. code::

                mpiexec -n 4 python test_allgatherv.py

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


    - ``rebalance(data, dests=None, random=False, random_seed=None, parallel=False)`` : manually redistribute data evenly across [selected] ranks.

        Arguments:

            ``data``: data to rebalance.

            ``dests``: selected ranks to distribute data to. By default, distribution includes all ranks.

            ``random``: flag to randomize order of the rows of the data. Default: `False`.

            ``random_seed``: number to initialze random number generator.

            ``parallel``: flag to indicate whether data is distributed. Default: `False`. Inside JIT default value depends on Bodo's distribution analysis algorithm for the data passed (For more information, see Data Distribution section below).

        .. code:: 

            import bodo
            import pandas as pd

            @bodo.jit
            def mean_power():
                df = pd.read_parquet("data/cycling_dataset.pq")
                df = df.sort_values("power")[df["power"] > 400]
                return df

            df = mean_power()
            print(df.shape)
            df = bodo.rebalance(df, parallel=True)
            print("After rebalance: ", df.shape)

        Save code in ``test_rebalance.py`` file and run with **mpiexec**.

        .. code::

                mpiexec -n 4 python test_rebalance.py

        .. parsed-literal::
            [stdout:0]
            (5, 10)
            After rebalance: (33, 10)
            [stdout:1]
            (18, 10)
            After rebalance: (33, 10)
            [stdout:2]
            (82, 10)
            After rebalance: (33, 10)
            [stdout:3]
            (26, 10)
            After rebalance: (32, 10)

        Example to distribute the data from all ranks to subset of ranks using ``dests`` argument.

        .. code:: 

            import bodo
            import pandas as pd

            @bodo.jit
            def mean_power():
                df = pd.read_parquet("data/cycling_dataset.pq")
                df = df.sort_values("power")[df["power"] > 400]
                return df

            df = mean_power()
            print(df.shape)
            df = bodo.rebalance(df, dests=[1,3], parallel=True)
            print("After rebalance: ", df.shape)

        Save code in ``test_rebalance.py`` file and run with **mpiexec**.

        .. code::

                mpiexec -n 4 python test_rebalance.py

        .. parsed-literal::
            [stdout:0]
            (5, 10)
            After rebalance: (0, 10)
            [stdout:1]
            (18, 10)
            After rebalance: (66, 10)
            [stdout:2]
            (82, 10)
            After rebalance: (0, 10)
            [stdout:3]
            (26, 10)
            After rebalance: (65, 10)

    - ``random_shuffle(data, seed=None, dests=None, parallel=False)`` : manually shuffle data evenly across [selected] ranks.

        Arguments:

            ``data``: data to shuffle.

            ``seed``: number to initialze random number generator.

            ``dests``: selected ranks to distribute shuffled data to. By default, distribution includes all ranks.

            ``parallel``: flag to indicate whether data is distributed. Default: `False`. Inside JIT default value depends on Bodo's distribution analysis algorithm for the data passed (For more information, see Data Distribution section below).

        .. code:: 

            import bodo
            import pandas as pd

            @bodo.jit
            def test_random_shuffle():
                df = pd.DataFrame({"A": range(100)})
                return df

            df = test_random_shuffle()
            print(df.head())
            df = bodo.random_shuffle(res, parallel=True)
            print(df.head())

        Save code in ``test_random_shuffle.py`` file and run with **mpiexec**.

        .. code::

                mpiexec -n 4 python test_random_shuffle.py

        .. parsed-literal::
            [stdout:1]
                A
            0  25
            1  26
            2  27
            3  28
            4  29
                A
            19  19
            10  10
            17  42
            9    9
            17  17
            [stdout:3]
                A
            0  75
            1  76
            2  77
            3  78
            4  79
                A
            6   31
            0   25
            24  49
            22  22
            5   30
            [stdout:2]
                A
            0  50
            1  51
            2  52
            3  53
            4  54
                A
            11  36
            24  24
            15  65
            14  14
            10  35
            [stdout:0]
                A
            0  0
            1  1
            2  2
            3  3
            4  4
                A
            4   29
            18  18
            8   58
            15  15
            3   28

    .. note::

        ``scatterv``, ``gatherv``, ``allgatherv``, ``rebalance``, and ``random_shuffle`` work with all distributable data types. This includes:
          * All supported numpy array types.
          * All supported pandas array types (with the exception of Interval Arrays).
          * All supported pandas Series types.
          * All supported DataFrame types.
          * All supported Index types (with the exception of Interval Index).
          * Tuples of the above types.

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

.. code:: 

    import bodo
    import pandas as pd
    @bodo.jit
    def mean_power_speed():
        df = pd.read_parquet("data/cycling_dataset.pq")
        m = df[["power", "speed"]].mean()
        return m

    res = mean_power_speed()
    print(res)

Save code in mean_power_speed.py and run it with **mpiexec** as follows:

.. code:: console

    mpiexec -n 4 python mean_power_speed.py

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
the previous example’s distributions by adding following line to `mean_power_speed` script:

.. code::console

    mean_power_speed.distributed_diagnostics()

.. code::console
    python mean_power_speed.py


.. parsed-literal::

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


Variables are renamed due to optimization. The output shows that
``power`` and ``speed`` columns of ``df`` are distributed (``1D_Block``)
but ``m`` is replicated (``REP``). This is because ``df`` is output of
``read_parquet`` and input of ``mean``, both of which can be distributed
by Bodo. ``m`` is output of ``mean``, which is always replicated
(available on every process).
