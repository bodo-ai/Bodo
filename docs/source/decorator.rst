.. _decorator:

@bodo.jit Decorator
===================

JIT Workflow
~~~~~~~~~~~~

Bodo provides a just-in-time (JIT) compilation workflow
using the `@bodo.jit` decorator.
It replaces the decorated Python functions with an optimized and parallelized
native binary version using advanced compilation methods.

Bodo uses `Numba <http://numba.pydata.org/>`_ for Python JIT workflow.
The decorated function is replaced with a dispatcher object,
which is compiled the first time it is called with a new combination of
argument types.
For example, the program below prints the dispatcher `f`,
and calls it twice with an integer as input::

    @bodo.jit
    def f(n):
        return pd.DataFrame({'A': np.arange(n)}).head(3)

    print(f)
    print(f(10))
    print(f(15))

The dispatcher will compile the function when it is called first,
but the second call will reuse the compiled binary. The output is as follows::

    CPUDispatcher(<function f at 0x122b56560>)
       A
    0  0
    1  1
    2  2
       A
    0  0
    1  1
    2  2

The `jit` decorators can be disabled if necessary (see :ref:`disable-jit`).

`distributed` Flag
~~~~~~~~~~~~~~~~~~

Bodo assumes argument and return variables to jitted functions are
:ref:`replicated <distribution>`.
However, the `distributed` flag can be used to indicate
distributed data. For example, the variable `df` is replicated
in the following program::

    @bodo.jit
    def example(n):
        df = pd.DataFrame({'A': np.arange(n)})
        return df

    print(example(1000))

The output indicates that the code is not parallelized::

    $ mpiexec -n 2 python ../example.py
    /Users/ehsan/dev/bodo/bodo/transforms/distributed_analysis.py:240: BodoWarning: No parallelism found for function 'example'. This could be due to unsupported usage. See distributed diagnostics for more information.
    "information.".format(self.func_ir.func_id.func_name)
           A
    0      0
    1      1
    2      2
    3      3
    4      4
    ..   ...
    995  995
    996  996
    997  997
    998  998
    999  999

    [1000 rows x 1 columns]
           A
    0      0
    1      1
    2      2
    3      3
    4      4
    ..   ...
    995  995
    996  996
    997  997
    998  998
    999  999

    [1000 rows x 1 columns]


Using the distributed flag allows the function to be parallelized and
distributed chunks returned::


    @bodo.jit(distributed={'df'})
    def example(n):
        df = pd.DataFrame({'A': np.arange(n)})
        return df

    print(example(1000))

The output indicates that the code is parallelized::

    $ mpiexec -n 2 python ../example.py
           A
    0      0
    1      1
    2      2
    3      3
    4      4
    ..   ...
    495  495
    496  496
    497  497
    498  498
    499  499

    [500 rows x 1 columns]
           A
    0    500
    1    501
    2    502
    3    503
    4    504
    ..   ...
    495  995
    496  996
    497  997
    498  998
    499  999

    [500 rows x 1 columns]


Type Annotation
~~~~~~~~~~~~~~~

Type annotation of some variables may be required when the
compiler is not able to infer the type automatically.
For example, when the name of an input file is
not constant to let the compiler inspect the file in compilation time,
type annotation is necessary::

    @bodo.jit(locals={'df':{'A': bodo.float64[:],
                            'B': bodo.int64[:],
                      }})
    def pq_read(file_name):
        df = pd.read_parquet(file_name)
        return df

    df = pq_read('example.parquet')
