.. _supported:

User Guide
==========

Bodo supports a subset of Python that is commonly used for
data analytics and machine learning. This section describes this subset
and explains how parallelization is performed.
The supported data structures for parallel/distributed datasets
are `Numpy <https://numpy.org/>`_ arrays, and
`Pandas <https://pandas.pydata.org/>`_ Dataframe, Series and Index objects.


Automatic Parallelization
-------------------------

Bodo parallelizes programs automatically based on the `map-reduce` parallel
pattern. Put simply, this means the compiler analyzes the program to
determine whether each parallelizable data structure and operation should
be distributed or not. This analysis uses the semantics of operations as
the program below demonstrates::

    @bodo.jit
    def example_1D():
        f = h5py.File("data.h5", "r")
        A = f['A'][:]
        return np.sum(A)

This program reads a one-dimensional array called `A` from file and sums its
values. Array `A` is the output of an I/O operation and is input to `np.sum`.
Based on semantics of I/O and `np.sum`, Bodo determines that `A` can be
distributed since I/O can output a distributed array and `np.sum` can
take a distributed array as input.
In `map-reduce` terminology, `A` is output of a `map` operator and is input
to a `reduce` operator. Hence,
Bodo distributes `A` and all operations associated with `A`
(i.e. I/O and `np.sum`) and generates a parallel binary.
This binary replaces the `example_1D` function in the Python program
automatically.

Bodo can only analyze and parallelize the supported data-parallel operations of
Numpy and Pandas (listed in this manual).
Hence, only the supported operations can be
used for distributed datasets and computations.
The sequential computation on other data structures can be any code that
`Numba supports <http://numba.pydata.org/numba-doc/latest/index.html>`_.

.. _distribution:

Data Distribution
~~~~~~~~~~~~~~~~~~

Bodo chooses data distribution automatically.
Data is either distributed in one-dimensional block (called `1D_Block`) manner
among processors, or fully replicated (called `REP`) on all processors.
`1D_Block` means that processors own equal
chunks of each distributed array, DataFrame or Series,
except possibly the last processor.
Dataframes and multi-dimensional arrays are distributed along their
first dimension.
For example, chunks of rows are distributed for dataframes and 2D matrices.
The figure below illustrates the distribution of a 9-element
one-dimensional Numpy array, as well
as a 9 by 2 array, on three processors:

.. image:: ../figs/dist.jpg
    :height: 500
    :width: 500
    :scale: 60
    :alt: distribution of 1D array
    :align: center


Argument and Return Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bodo assumes argument and return variables to jitted functions are
replicated. However, the user can annotate these variables to indicate
distributed data. In this case,
the user is responsible for handling of the distributed data chunks outside
the Bodo scope. For example, the data can come from other jitted functions::

    @bodo.jit(distributed={'A'})
    def example_return(n):
        A = np.arange(n)
        return A

    @bodo.jit(distributed={'B'})
    def example_arg(B):
        return B.sum()

    n = 10000
    A = example_return(n)
    print(A)
    s = example_arg(A)
    print(s)


Here is the output of this example when run on two cores::

    $ mpiexec -n 2 python ../example.py
    [   0    1    2 ... 4997 4998 4999]
    [5000 5001 5002 ... 9997 9998 9999]
    49995000
    49995000

In each process, `example_return` returns a chunk of `A` and `example_arg`
receives the same chunk. Data chunks can also be transformed before passing
back to Bodo, but the data set size cannot change since
Bodo expects equal chunks on different cores.

The `distributed` flag is only applicable to input argument variables
and return variables currently. In addition, only distributable data structures
are allowed (e.g. dataframes and arrays, but not lists).


Distribution Report
~~~~~~~~~~~~~~~~~~~

The distributions found by Bodo can be printed either by setting
`BODO_DISTRIBUTED_DIAGNOSTICS=1` or calling `distributed_diagnostics()`
on the compiled function. For example, consider example code below::

    @bodo.jit
    def example_1D():
        f = h5py.File("data.h5", "r")
        A = f['A'][:]
        return A.sum()

    example_1D()
    example_1D.distributed_diagnostics()

Here is the diagnostics output::

    Distributed diagnostics for function example_1D, example.py (7)

    Data distributions:
    $A.39.101            1D_Block

    Parfor distributions:
    0                    1D_Block

    Distributed listing for function example_1D, example.py (7)
    -----------------------------------------------------| parfor_id/variable: distribution
    @bodo.jit                                            |
    def example_1D():                                    |
        f = h5py.File("bodo/tests/data/lr.hdf5", "r")    |
        A = f['A'][:]                            |
        return A.sum()-----------------------------------| #0: 1D_Block


This report suggests that the function has an array that is distributed in
`1D_Block` fashion. The variable name is renamed from `A` to `$A.39.101`
through the optimization passes. The report also lists a
`parfor` (data-parallel for loop) that is also `1D_Block` distributed.
Parfor number 0 is generated by the compiler from the `sum` operation
and can be distributed since its associated data (`A`) can be distributed.


There is also a form of one-dimensional block distribution called
`1D_Block_Var`, which indicates that distributed chunks may not have
equal sizes across processors. For example, a dataframe filter
operation can produce unequal data chunks depending on input values on
different processors. This extra piece of information may be useful for
advanced users (e.g. passing data chunks to other systems).


Explicit Parallel Loops
-----------------------

Sometimes explicit parallel loops are required since a program cannot be
written in terms of data-parallel operators easily.
In this case, one can use Bodo's ``prange`` in place of ``range`` to specify
that a loop can be parallelized. The user is required to make sure the
loop does not have cross iteration dependencies except for supported
reductions.

The example below demonstrates a parallel loop with a reduction::

    from bodo import jit, prange
    @jit
    def prange_test(n):
        A = np.random.ranf(n)
        s = 0
        for i in prange(len(A)):
            s += A[i]
        return s

Currently, reductions using ``+=``, ``*=``, ``min``, and ``max`` operators are
supported.


File I/O
--------

Bodo automatically parallelizes I/O of different nodes in a distributed setting
without any code changes.

Supported formats
~~~~~~~~~~~~~~~~~

Currently, Bodo supports I/O for `Parquet <http://parquet.apache.org/>`_,
CSV, `HDF5 <http://www.h5py.org/>`_ and Numpy binaries formats.

.. _parquet-section:

Parquet
^^^^^^^

For Parquet, the syntax is the same as ``Pandas``:
``pd.read_parquet(path)``, where path can be a parquet file or a directory with multiple parquet files 
(all are part of the same dataframe)::

    @bodo.jit
    def example_pq():
        df = pd.read_parquet('example.pq')

``to_parquet(name)`` with distributed data writes to a 'folder' called ``name``.
Each process writes one file into the folder, but if the data is not distributed,
``to_parquet(name)`` writes to a file called ``name``:: 

    df = pd.DataFrame({'A': np.arange(n)})

    @bodo.jit
    def example1_pq(df):
        df.to_parquet('example1.pq')

    @bodo.jit(distributed={'df'})
    def example2_pq(df):
        df.to_parquet('example2.pq')

    if bodo.get_rank() == 0:
        example1_pq(df)
    example2_pq(df)

Run the code above with 4 processors::

    $ mpiexec -n 4 python ../example_pq.py

``example1_pq(df)`` writes 1 single file, and ``example2_pq(df)`` writes a folder containing 4 parquet files::

    .
    ├── example1.pq
    ├── example2.pq
    │   ├── part-00.parquet
    │   ├── part-01.parquet
    │   ├── part-02.parquet
    │   └── part-03.parquet

.. _csv-section:

CSV
^^^
For csv, the syntax is the also same as Pandas::

    @bodo.jit
    def example_csv():
        df = pd.read_csv('example.csv')

``to_csv()`` always writes to a single file, regardless of the number
of processes and whether the data is distributed::

    df = pd.DataFrame({'A': np.arange(n)})

    @bodo.jit
    def example1_csv(df):
        df.to_csv('example1.csv')

    @bodo.jit(distributed={'df'})
    def example2_csv(df):
        df.to_csv('example2.csv')

    if bodo.get_rank() == 0:
        example1_csv(df)
    example2_csv(df)

Run the code above with 4 processors::

    $ mpiexec -n 4 python ../example_csv.py

each ``example1_csv(df)`` and ``example2_csv(df)`` writes to a single file::

    .
    ├── example1.csv
    ├── example2.csv

HDF5
^^^^

For HDF5, the syntax is the same as the `h5py <http://www.h5py.org/>`_ package.
For example::

    @bodo.jit
    def example_h5():
        f = h5py.File("data.hdf5", "r")
        X = f['points'][:]
        Y = f['responses'][:]

Numpy binaries
^^^^^^^^^^^^^^

Numpy's ``fromfile`` and ``tofile`` are supported as below::

    @bodo.jit
    def example_np_io():
        A = np.fromfile("myfile.dat", np.float64)
        ...
        A.tofile("newfile.dat")

Input array types
~~~~~~~~~~~~~~~~~

Bodo needs to know the types of input arrays. If the file name is a constant
string or function argument, Bodo tries to look at the file at compile time
and recognize the types.
Otherwise, the user is responsible for providing the types similar to
`Numba's typing syntax
<http://numba.pydata.org/numba-doc/latest/reference/types.html>`_. For
example::

    @bodo.jit(locals={'df':{'one': bodo.float64[:],
                      'two': bodo.string_array_type,
                      'three': bodo.bool_[:],
                      'four': bodo.float64[:],
                      'five': bodo.string_array_type,
                      }})
    def example_df_schema(fname1, fname2, flag):
        if flag:
            file_name = fname1
        else:
            file_name = fname2
        df = pd.read_parquet(file_name)


     @bodo.jit(locals={'X': bodo.float64[:,:], 'Y': bodo.float64[:]})
     def example_h5(fname1, fname2, flag):
        if flag:
            file_name = fname1
        else:
            file_name = fname2
         f = h5py.File(file_name, "r")
         X = f['points'][:]
         Y = f['responses'][:]

Amazon S3
~~~~~~~~~

Reading :ref:`csv <csv-section>`, reading and writing :ref:`Parquet <parquet-section>` files from and to Amazon S3 is supported. 
The ``s3fs`` package must be available, and the file path should start with :code:`s3://`::

    @bodo.jit
    def example_s3_csv():
        df = pd.read_csv('s3://bucket-name/file_name.csv')

These environment variables are used for File I/O with S3 credentials:
  - ``AWS_ACCESS_KEY_ID``
  - ``AWS_SECRET_ACCESS_KEY``
  - ``AWS_DEFAULT_REGION``: default as ``us-east-1``
  - ``AWS_S3_ENDPOINT``: specify custom host name, default as AWS endpoint(``s3.amazonaws.com``)


Hadoop Distributed File System (HDFS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reading :ref:`csv <csv-section>`, reading and writing:ref:`Parquet <
parquet-section>` from and to Hadoop Distributed File System (HDFS) is supported. 
The file path should start with ``hdfs://``::

    @bodo.jit
    def example_hdfs_parquet():
        df = pd.read_parquet('hdfs://host:port/dir/file_name.pq')

These environment variables are used for File I/O with HDFS:
  - ``HADOOP_HOME``: the root of your installed Hadoop distribution. Often has `lib/native/libhdfs.so`.
  - ``ARROW_LIBHDFS_DIR``: location of libhdfs. Often as ``$HADOOP_HOME/lib/native``.
  - ``CLASSPATH``: must contain the Hadoop jars. You can set these using::

        export CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob`

``$HADOOP_HOME/etc/hadoop/hdfs-site.xml`` provides default behaviors for the HDFS client.
For example, the following configuration has default block replication set as 3::
    
    <configuration>
        <property>
            <name>dfs.replication</name>
            <value>3</value>
        </property>
    </configuration>



Print
-----

Bodo avoids repeated prints of replicated data by printing
them only once (on rank 0) instead of one print per process.
However, chunks of distributed data are
printed on all processes as expected.
Programmers can use `bodo.parallel_print` for printing
replicated data on all processes if desired.


Parallel APIs
-------------

Bodo provides a limited number of parallel APIs to
support advanced cases that may need them (the reference communicator is `MPI_COMM_WORLD`).

* :func:`bodo.get_rank` Get the rank of the process (same as `MPI_Comm_rank`).
* :func:`bodo.get_size` Get the number of processes (same as `MPI_Comm_size`).
* :func:`bodo.barrier` Blocks until all processes have reached this call
  (same as `MPI_Barrier`).
* :func:`bodo.send` Blocking send of data from a process (same as `MPI_SEND`)
* :func:`bodo.recv` Blocking recv of data from a process (same as `MPI_RECV`)
* :func:`bodo.isend` Asynchronous send of data from a process (same as `MPI_ISEND`)
* :func:`bodo.irecv` Asynchronous recv of data from a process (same as `MPI_IRECV`)
* :func:`bodo.gatherv` Gathers all data chunks into process 0
  (same as `MPI_Gatherv`).
* :func:`bodo.allgatherv` Gathers all data chunks and delivers to all processes
  (same as `MPI_Allgatherv`).


Regular Expressions Support
---------------------------

Bodo supports regular expressions using `Python's standard re library <https://docs.python.org/3/library/re.html>`_.
All functions and attributes except `finditer()` are supported.
However, cases where the output could be `None` to designate unmatched
groups are not supported yet. The APIs where this case is possible are
`Match.group()`, `Match.groups()`, `Match.groupdict()`, `Match.lastindex`
and `Match.lastgroup`.
