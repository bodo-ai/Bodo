.. _supported:

User Guide
==========

Bodo supports a subset of Python that is commonly used for
data analytics and machine learning. This section describes this subset
and explains how parallelization is performed.
The supported data structures for parallel/distributed datasets
are `Numpy <http://www.numpy.org/>`_ arrays, and
`Pandas <http://pandas.pydata.org/>`_ Dataframe, Series and Index objects.


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
the user is responsile for handling of the distributed data chunks outside
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

Currently, Bodo supports I/O for `Parquet <http://parquet.apache.org/>`_,
CSV, Numpy binaries and `HDF5 <http://www.h5py.org/>`_ formats.

For Parquet and CSV, the syntax is the same as Pandas::

    @bodo.jit
    def example_pq():
        df = pd.read_parquet('example.pq')


Reading CSV and Parquet files from Amazon S3 is also supported.
The file path should start with `s3://`, and the `s3fs` package should
be available.
Currently, specifying the region using the `AWS_DEFAULT_REGION` environment
variable is required.


For HDF5, the syntax is the same as the `h5py <http://www.h5py.org/>`_ package.
For example::

    @bodo.jit
    def example_h5():
        f = h5py.File("data.hdf5", "r")
        X = f['points'][:]
        Y = f['responses'][:]

Numpy's `fromfile` and `tofile` are supported as below::

    @bodo.jit
    def example_np_io():
        A = np.fromfile("myfile.dat", np.float64)
        ...
        A.tofile("newfile.dat")


Bodo automatically parallelizes I/O of different nodes in a distributed setting
without any code changes.

Bodo needs to know the types of input arrays. If the file name is a constant
string or function argument, Bodo tries to look at the file at compile time
and recognize the types.
Otherwise, the user is responsile for providing the types similar to
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

Print
-----

Bodo assigns `REP` to distributable arguments of ``print`` functions
(to make sure values are identical on all processors) and prints values
only once instead of one print per process. The intention is to avoid
unexpected behavior, especially when running on large number of processors.
Programmers can use `bodo.parallel_print` for printing distributed chunks of
data or parallel prints of other values.


Parallel APIs
-------------

Bodo provides a limited number of parallel APIs to
support advanced cases that may need them.

* :func:`bodo.get_rank` Get the rank of the process (same as `MPI_Comm_rank`).
* :func:`bodo.get_size` Get the number of processes (same as `MPI_Comm_size`).
* :func:`bodo.barrier` Blocks until all processes have reached this call
  (same as `MPI_Barrier`).
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
