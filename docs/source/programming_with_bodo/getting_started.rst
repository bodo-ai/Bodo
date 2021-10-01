Getting Started
=============================

This section provides a short tutorial that covers the basics of using
Bodo and explains its important concepts.


Parallel Pandas with Bodo
-------------------------

First, we demonstrate how Bodo automatically parallelizes and optimizes
standard Python programs that make use of pandas and NumPy, without the
need to rewrite your code. Bodo can scale your analytics code to
thousands of cores, providing orders of magnitude speed up depending on
program characteristics.

Generate data
~~~~~~~~~~~~~

To begin, let’s generate a simple dataset (the size of this dataframe in
memory is approximately 305 MB, and the size of the written Parquet file
is 77 MB):

.. code:: ipython3

    import pandas as pd
    import numpy as np
    
    NUM_GROUPS = 30
    NUM_ROWS = 20_000_000
    df = pd.DataFrame({
        "A": np.arange(NUM_ROWS) % NUM_GROUPS,
        "B": np.arange(NUM_ROWS)
    })
    df.to_parquet("data/example1.pq")
    print(df)


.. parsed-literal::

               A         B
    0          0         0
    1          1         1
    2          2         2
    3          3         3
    4          4         4
    ...       ..       ...
    19999995  15  19999995
    19999996  16  19999996
    19999997  17  19999997
    19999998  18  19999998
    19999999  19  19999999
    
    [20000000 rows x 2 columns]


Data Analysis
~~~~~~~~~~~~~

Now let’s read and process this dataframe. First using Python and
pandas:

.. code:: ipython3

    def test():
        df = pd.read_parquet("data/example1.pq")
        df2 = df.groupby("A").sum()
        m = df2.B.max()
        print(m)
    
    test()


.. parsed-literal::

    6666676000003


Now let’s run it with Bodo in parallel. To do this, all that we have to
do is add the ``bodo.jit`` decorator to the function, and run the
program with MPI (on Jupyter Notebook, use the ``%%px``
`magic <https://ipyparallel.readthedocs.io/en/latest/magics.html>`__ to
run on MPI engines):

.. code:: ipython3

    import pandas as pd
    import bodo
    
    @bodo.jit
    def test():
        df = pd.read_parquet("data/example1.pq")
        df2 = df.groupby("A").sum()
        m = df2.B.max()
        print(m)
    
    test()


.. parsed-literal::

    [stdout:0] 6666676000003


Although the program appears to be a regular sequential Python program,
Bodo compiles and *transforms* the decorated code (the ``test`` function
in this example) under the hood, so that it can run in parallel on many
cores. Each core operates on a different chunk of the data and
communicates with other cores when necessary.

Parallel Python Processes
~~~~~~~~~~~~~~~~~~~~~~~~~

With Bodo, all processes are running the same code. Bodo manages
parallelism inside ``jit`` functions to match sequential Python as much
as possible. On the other hand, the code outside ``jit`` functions runs
as regular Python on all processes. For example, the code below when run
on 4 processes produces 4 prints, one for each Python process:

.. code:: ipython3

    
    @bodo.jit
    def test():
        df = pd.read_parquet("data/example1.pq")
        df2 = df.groupby("A").sum()
        m = df2.B.max()
        return m
    
    m = test()
    print(m)


.. parsed-literal::

    [stdout:0] 6666676000003
    [stdout:1] 6666676000003
    [stdout:2] 6666676000003
    [stdout:3] 6666676000003


Prints
~~~~~~

Bodo prints replicated values like ``m`` only once (on process ``0``) to
avoid redundant printing, but we can use ``bodo.parallel_print`` to see
prints on all processes:

.. code:: ipython3

    
    @bodo.jit
    def test():
        df = pd.read_parquet("data/example1.pq")
        df2 = df.groupby("A").sum()
        m = df2.B.max()
        bodo.parallel_print(m)
    
    test()


.. parsed-literal::

    [stdout:0] 6666676000003
    [stdout:1] 6666676000003
    [stdout:2] 6666676000003
    [stdout:3] 6666676000003


Parallel Data Read
~~~~~~~~~~~~~~~~~~

Bodo can read data from storage such as Parquet files in parallel. This
means that each process reads only its own chunk of data (which can be
proportionally faster than sequential read). The example below
demonstrates parallel read by printing data chunks on different cores:

.. code:: ipython3

    
    @bodo.jit
    def test():
        df = pd.read_parquet("data/example1.pq")
        print(df)
    
    test()


.. parsed-literal::

    [stdout:0] 
              A        B
    0         0        0
    1         1        1
    2         2        2
    3         3        3
    4         4        4
    ...      ..      ...
    4999995  15  4999995
    4999996  16  4999996
    4999997  17  4999997
    4999998  18  4999998
    4999999  19  4999999
    
    [5000000 rows x 2 columns]
    [stdout:1] 
              A        B
    5000000  20  5000000
    5000001  21  5000001
    5000002  22  5000002
    5000003  23  5000003
    5000004  24  5000004
    ...      ..      ...
    9999995   5  9999995
    9999996   6  9999996
    9999997   7  9999997
    9999998   8  9999998
    9999999   9  9999999
    
    [5000000 rows x 2 columns]
    [stdout:2] 
               A         B
    10000000  10  10000000
    10000001  11  10000001
    10000002  12  10000002
    10000003  13  10000003
    10000004  14  10000004
    ...       ..       ...
    14999995  25  14999995
    14999996  26  14999996
    14999997  27  14999997
    14999998  28  14999998
    14999999  29  14999999
    
    [5000000 rows x 2 columns]
    [stdout:3] 
               A         B
    15000000   0  15000000
    15000001   1  15000001
    15000002   2  15000002
    15000003   3  15000003
    15000004   4  15000004
    ...       ..       ...
    19999995  15  19999995
    19999996  16  19999996
    19999997  17  19999997
    19999998  18  19999998
    19999999  19  19999999
    
    [5000000 rows x 2 columns]


Looking at column B, we can clearly see that each process has a separate
chunk of the original dataframe.

Parallelizing Computation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: img/groupby.jpg
   :alt: Groupby shuffle communication pattern

   Groupby shuffle communication pattern

Bodo parallelizes computation automatically by dividing the work between
cores and performing the necessary data communication. For example, the
``groupby`` operation in our example needs the data of each group to be
on the same processor. This requires *shuffling* data across the
cluster. Bodo uses
`MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`__ for
efficient communication, which is usually much faster than alternative
methods.

Parallel Write
~~~~~~~~~~~~~~

Bodo can write data to storage in parallel as well:

.. code:: ipython3

    
    @bodo.jit
    def test():
        df = pd.read_parquet("data/example1.pq")
        df2 = df.groupby("A").sum()
        df2.to_parquet("data/example1-df2.pq")
    
    test()

Now let’s read and print the results with pandas:

.. code:: ipython3

    import pandas as pd
    
    df = pd.read_parquet("data/example1-df2.pq")
    print(df)


.. parsed-literal::

                    B
    A                
    0   6666663333330
    4   6666665999998
    6   6666667333332
    16  6666674000002
    20  6666656666670
    24  6666659333334
    28  6666661999998
    1   6666663999997
    7   6666667999999
    8   6666668666666
    11  6666670666667
    12  6666671333334
    13  6666672000001
    15  6666673333335
    18  6666675333336
    5   6666666666665
    19  6666676000003
    21  6666657333336
    22  6666658000002
    23  6666658666668
    29  6666662666664
    2   6666664666664
    3   6666665333331
    9   6666669333333
    10  6666670000000
    14  6666672666668
    17  6666674666669
    25  6666660000000
    26  6666660666666
    27  6666661333332


The order of the ``groupby`` results generated by Bodo can differ from
pandas since Bodo doesn’t automatically sort the output distributed data
(it is expensive and not necessary in many cases). Users can explicitly
sort dataframes at any point if desired.

Specifying Data Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bodo automatically distributes data and computation in Bodo functions by
analyzing them for parallelization. However, Bodo does not know how
input parameters of Bodo functions are distributed, and similarly how
the user wants to handle return values. As such, Bodo assumes that input
parameters and return values are *replicated* by default, meaning that
every process receives the same input data and returns the same output,
as opposed to different data chunks.

.. warning::

    The distribution scheme of input parameters and return values determines the distribution scheme for variables inside the Bodo function that depend on them.


To illustrate this effect, let’s return the ``groupby`` output from the
Bodo function:

.. code:: ipython3

    import pandas as pd
    import bodo
    
    @bodo.jit
    def test():
        df = pd.read_parquet("data/example1.pq")
        df2 = df.groupby("A").sum()
        return df2
    
    df2 = test()
    print(df2)


.. parsed-literal::

    [stdout:0] 
                    B
    A                
    0   6666663333330
    1   6666663999997
    2   6666664666664
    3   6666665333331
    4   6666665999998
    5   6666666666665
    6   6666667333332
    7   6666667999999
    8   6666668666666
    9   6666669333333
    10  6666670000000
    11  6666670666667
    12  6666671333334
    13  6666672000001
    14  6666672666668
    15  6666673333335
    16  6666674000002
    17  6666674666669
    18  6666675333336
    19  6666676000003
    20  6666656666670
    21  6666657333336
    22  6666658000002
    23  6666658666668
    24  6666659333334
    25  6666660000000
    26  6666660666666
    27  6666661333332
    28  6666661999998
    29  6666662666664
    [stdout:1] 
                    B
    A                
    0   6666663333330
    1   6666663999997
    2   6666664666664
    3   6666665333331
    4   6666665999998
    5   6666666666665
    6   6666667333332
    7   6666667999999
    8   6666668666666
    9   6666669333333
    10  6666670000000
    11  6666670666667
    12  6666671333334
    13  6666672000001
    14  6666672666668
    15  6666673333335
    16  6666674000002
    17  6666674666669
    18  6666675333336
    19  6666676000003
    20  6666656666670
    21  6666657333336
    22  6666658000002
    23  6666658666668
    24  6666659333334
    25  6666660000000
    26  6666660666666
    27  6666661333332
    28  6666661999998
    29  6666662666664
    [stdout:2] 
                    B
    A                
    0   6666663333330
    1   6666663999997
    2   6666664666664
    3   6666665333331
    4   6666665999998
    5   6666666666665
    6   6666667333332
    7   6666667999999
    8   6666668666666
    9   6666669333333
    10  6666670000000
    11  6666670666667
    12  6666671333334
    13  6666672000001
    14  6666672666668
    15  6666673333335
    16  6666674000002
    17  6666674666669
    18  6666675333336
    19  6666676000003
    20  6666656666670
    21  6666657333336
    22  6666658000002
    23  6666658666668
    24  6666659333334
    25  6666660000000
    26  6666660666666
    27  6666661333332
    28  6666661999998
    29  6666662666664
    [stdout:3] 
                    B
    A                
    0   6666663333330
    1   6666663999997
    2   6666664666664
    3   6666665333331
    4   6666665999998
    5   6666666666665
    6   6666667333332
    7   6666667999999
    8   6666668666666
    9   6666669333333
    10  6666670000000
    11  6666670666667
    12  6666671333334
    13  6666672000001
    14  6666672666668
    15  6666673333335
    16  6666674000002
    17  6666674666669
    18  6666675333336
    19  6666676000003
    20  6666656666670
    21  6666657333336
    22  6666658000002
    23  6666658666668
    24  6666659333334
    25  6666660000000
    26  6666660666666
    27  6666661333332
    28  6666661999998
    29  6666662666664


.. parsed-literal::

    [stderr:0] 
    /Users/ehsan/dev/bodo/bodo/transforms/distributed_analysis.py:229: BodoWarning: No parallelism found for function 'test'. This could be due to unsupported usage. See distributed diagnostics for more information.
      warnings.warn(


As we can see, ``df2`` has the same data on every process. Furthermore,
Bodo warns that it didn’t find any parallelism inside the ``test``
function. In this example, every process reads the whole input Parquet
file and executes the same sequential program. The reason is that Bodo
makes sure all variables dependent on ``df2`` have the same
distribution, creating an inverse cascading effect.

``distributed`` Flag
~~~~~~~~~~~~~~~~~~~~

The user can tell Bodo what input/output variables should be distributed
using the ``distributed`` flag:

.. code:: ipython3

    
    @bodo.jit(distributed=["df2"])
    def test():
        df = pd.read_parquet("data/example1.pq")
        df2 = df.groupby("A").sum()
        return df2
    
    df2 = test()
    print(df2)


.. parsed-literal::

    [stdout:0] 
                    B
    A                
    0   6666663333330
    4   6666665999998
    6   6666667333332
    16  6666674000002
    20  6666656666670
    24  6666659333334
    28  6666661999998
    [stdout:1] 
                    B
    A                
    1   6666663999997
    7   6666667999999
    8   6666668666666
    11  6666670666667
    12  6666671333334
    13  6666672000001
    15  6666673333335
    18  6666675333336
    [stdout:2] 
                    B
    A                
    5   6666666666665
    19  6666676000003
    21  6666657333336
    22  6666658000002
    23  6666658666668
    29  6666662666664
    [stdout:3] 
                    B
    A                
    2   6666664666664
    3   6666665333331
    9   6666669333333
    10  6666670000000
    14  6666672666668
    17  6666674666669
    25  6666660000000
    26  6666660666666
    27  6666661333332


In this case, the program is fully parallelized and chunks of data are
returned to Python on different processes.

Basic benchmarking of the pandas example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now let’s do some basic benchmarking to observe the effect of Bodo’s
automatic parallelization. Here we are only scaling up to a few cores,
but Bodo can scale the same code to thousands of cores in a cluster.

Let’s add timers and run the code again with pandas:

.. code:: ipython3

    import pandas as pd
    import time
    
    def test():
        df = pd.read_parquet("data/example1.pq")
        t0 = time.time()
        df2 = df.groupby("A").sum()
        m = df2.B.max()
        print("Compute time:", time.time() - t0, "secs")
        return m
    
    result = test()


.. parsed-literal::

    Compute time: 0.46109819412231445 secs


Now let’s measure Bodo’s execution time.

.. code:: ipython3

    import time
    
    @bodo.jit
    def test():
        df = pd.read_parquet("data/example1.pq")
        t0 = time.time()
        df2 = df.groupby("A").sum()
        m = df2.B.max()
        print("Compute time:", time.time() - t0, "secs")
        return m
    
    result = test()


.. parsed-literal::

    [stdout:0] Compute time: 0.22473560000071302 secs


As we can see, Bodo computes results faster than pandas using parallel
computation. The speedup depends on the data and program
characteristics, as well as the number of cores used. Usually, we can
continue scaling to many more cores as long as the data is large enough.

Note how we included timers inside the Bodo function. This avoids
measuring compilation time since Bodo compiles each ``jit`` function the
first time it is called. Not measuring compilation time in benchmarking
is usually important since:

1. Compilation time is often not significant for large computations in
   real settings but simple benchmarks are designed to run quickly
2. Functions can potentially be compiled and cached ahead of execution
   time
3. Compilation happens only once but the same function may be called
   multiple times, leading to inconsistent measurements

Pandas User-Defined Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

User-defined functions (UDFs) offer significant flexibility but have
high overhead in Pandas. Bodo can accelerate UDFs significantly,
allowing flexibility without performance overheads. Let’s modify our
example to use UDFs and measure performance again:

.. code:: ipython3

    def test():
        df = pd.read_parquet("data/example1.pq")
        t0 = time.time()
        df2 = df.groupby("A")["B"].agg((lambda a: (a==1).sum(), lambda a: (a==2).sum(), lambda a: (a==3).sum()))
        m = df2.mean()
        print("Compute time:", time.time() - t0, "secs")
        return m
    
    result = test()


.. parsed-literal::

    Compute time: 3.1591920852661133 secs


Running this example with Bodo is significantly faster, even on a single
core:

.. code:: ipython3

    import bodo
    
    @bodo.jit
    def test():
        df = pd.read_parquet("data/example1.pq")
        t0 = time.time()
        df2 = df.groupby("A")["B"].agg((lambda a: (a==1).sum(), lambda a: (a==2).sum(), lambda a: (a==3).sum()))
        m = df2.mean()
        print("Compute time:", time.time() - t0, "secs")
        return m
    
    result = test()


.. parsed-literal::

    Compute time: 0.8296882309950888 secs


Bodo’s parallelism improves performance further:

.. code:: ipython3

    
    @bodo.jit
    def test():
        df = pd.read_parquet("data/example1.pq")
        t0 = time.time()
        df2 = df.groupby("A")["B"].agg((lambda a: (a==1).sum(), lambda a: (a==2).sum(), lambda a: (a==3).sum()))
        m = df2.mean()
        print("Compute time:", time.time() - t0, "secs")
        return m
    
    result = test()


.. parsed-literal::

    [stdout:0] Compute time: 0.3215170180046698 secs


Memory Optimizations in Bodo
----------------------------

Bodo also improves performance by eliminating intermediate array values
in computations such as expressions in Pandas and Numpy. The Monte Carlo
Pi Estimation example demonstrates this effect:

.. code:: ipython3

    import numpy as np
    
    def calc_pi(n):
        t1 = time.time()
        x = 2 * np.random.ranf(n) - 1
        y = 2 * np.random.ranf(n) - 1
        pi = 4 * np.sum(x**2 + y**2 < 1) / n
        print("Execution time:", time.time()-t1, "\nresult:", pi)
    
    calc_pi(2 * 10**8)


.. parsed-literal::

    Execution time: 9.101144075393677 
    result: 3.14178548


Bodo is faster even on a single core since it avoids creating arrays
alltogether:

.. code:: ipython3

    @bodo.jit
    def calc_pi(n):
        t1 = time.time()
        x = 2 * np.random.ranf(n) - 1
        y = 2 * np.random.ranf(n) - 1
        pi = 4 * np.sum(x**2 + y**2 < 1) / n
        print("Execution time:", time.time()-t1, "\nresult:", pi)
    
    calc_pi(2 * 10**8)


.. parsed-literal::

    Execution time: 2.422189676988637 
    result: 3.14182726


Data-parallel array computations typically scale well too:

.. code:: ipython3

    import numpy as np
    
    @bodo.jit
    def calc_pi(n):
        t1 = time.time()
        x = 2 * np.random.ranf(n) - 1
        y = 2 * np.random.ranf(n) - 1
        pi = 4 * np.sum(x**2 + y**2 < 1) / n
        print("Execution time:", time.time()-t1, "\nresult:", pi)
    
    calc_pi(2 * 10**8)


.. parsed-literal::

    [stdout:0] 
    Execution time: 0.634156896994682 
    result: 3.14174714


Unsupported Pandas/Python Features
----------------------------------

Supported Pandas Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bodo supports a large subset of Pandas APIs as listed
`here <http://docs.bodo.ai/latest/source/pandas.html>`__. Moreover,
dataframe schemas (column names and types) should be stable in
operations. For example, key column names to ``group`` have to be
constant for output type to be stable. This example demonstrates the
issue:

.. code:: ipython3

    import bodo
    
    @bodo.jit(distributed=False)
    def f(a, i):
        column_list = a[:i]  # some computation that cannot be inferred statically
        df = pd.DataFrame({"A": [1, 2, 1], "B": [4, 5, 6]})
        return df.groupby(column_list).sum()
    
    a = ["A", "B"]
    i = 1
    f(a, i)


::


    ---------------------------------------------------------------------------

    BodoError                                 Traceback (most recent call last)

    <ipython-input-20-8ff85fad034d> in <module>
          9 a = ["A", "B"]
         10 i = 1
    ---> 11 f(a, i)
    

    ~/dev/bodo/bodo/numba_compat.py in _compile_for_args(***failed resolving arguments***)
        841         del args
        842         if error:
    --> 843             raise error
        844 
        845 


    BodoError: groupby(): 'by' parameter only supports a constant column label or column labels.
    
    File "<ipython-input-20-8ff85fad034d>", line 7:
    def f(a, i):
        <source elided>
        df = pd.DataFrame({"A": [1, 2, 1], "B": [4, 5, 6]})
        return df.groupby(column_list).sum()
        ^
    


The code can most often be refactored to compute the key list in regular
Python and pass as argument to Bodo:

.. code:: ipython3

    @bodo.jit(distributed=False)
    def f(column_list):
        df = pd.DataFrame({"A": [1, 2, 1], "B": [4, 5, 6]})
        return df.groupby(column_list).sum()
    
    a = ["A", "B"]
    i = 1
    column_list = a[:i]
    f(column_list)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>B</th>
        </tr>
        <tr>
          <th>A</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1</th>
          <td>10</td>
        </tr>
        <tr>
          <th>2</th>
          <td>5</td>
        </tr>
      </tbody>
    </table>
    </div>



Supported Python Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bodo relies on Numba for supporting basic Python features. Therefore,
Python constructs that are not supported by Numba (see Numba
documentation
`here <http://numba.pydata.org/numba-doc/latest/reference/pysupported.html>`__)
should be avoided in Bodo programs. For example:

-  context manager: ``with`` (except for ``with bodo.objmode``)
-  ``async`` features
-  ``set``, ``dict`` and ``generator`` comprehensions
-  List containing values of heterogeneous type

   -  myList = [1, 2]
      myList.append(“A”)

-  Dictionary containing values of heterogeneous type

   -  myDict = {“A”: 1}
      myDict[“B”] = “C”

Parallel Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~

Bodo can parallelize Pandas DataFrame and Series data structures, as
well as Numpy arrays. However, collections like lists, sets and
dictionaries cannot be parallelized yet.
