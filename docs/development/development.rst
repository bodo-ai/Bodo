.. _development:

Bodo Development
================

Technology Overview and Architecture
------------------------------------

This `slide deck <https://drive.google.com/file/d/1mHrbjAEfP6p-o-bWJOVdtmKNEA7lreDt/view?usp=sharing>`_
provides an overview of Bodo technology and software architecture.


Numba Development
-----------------

Bodo sits on top of Numba and is heavily tied to many of its features.
Therefore, understanding Numba's internal details and being able to
develop Numba extensions is necessary.


- Start with `basic overview of Numba use <http://numba.pydata.org/numba-doc/latest/user/5minguide.html>`_ and try the examples.
- `User documentation <http://numba.pydata.org/numba-doc/latest/user/index.html>`_ is generally helpful for overview of features.
- | `ParallelAccelerator documentation <http://numba.pydata.org/numba-doc/latest/user/parallel.html>`_
    provides overview of parallel analysis and transformations in Numba
    (also used in Bodo).
- `Setting up Numba for development <http://numba.pydata.org/numba-doc/latest/developer/contributing.html>`_
- | `Numba architecture page <http://numba.pydata.org/numba-doc/latest/developer/architecture.html>`_
    is a good starting point for understanding the internals.
- | The `overload guide page of Numba <http://numba.pydata.org/numba-doc/latest/extending/overloading-guide.html>`_
    is useful for understanding the process of implementing
    new functionality and specializing to data types.
- | Learning Numba IR is crucial for understanding transformations.
    See the `IR classes <https://github.com/numba/numba/blob/master/numba/ir.py>`_.
    Setting `NUMBA_DEBUG_ARRAY_OPT=1` shows the IR at different stages
    of ParallelAccelerator and Bodo transformations. Run `a simple parallel
    example <http://numba.pydata.org/numba-doc/latest/user/parallel.html#explicit-parallel-loops>`_
    and make sure you understad the IR at different stages.
- | `Exending Numba page <http://numba.pydata.org/numba-doc/latest/extending/index.html>`_
    provides details on how to provide native implementations for data types and functions.
    The low-level API should be avoided as much as possible for ease of development and
    code readability. The `unicode support <https://github.com/numba/numba/blob/master/numba/unicode.py>`_
    in Numba is an example of a modern extension for Numba (documentation planned).
- | A more complex extension is `the new dictionary implementation in
    Numba <https://github.com/numba/numba/blob/master/numba/dictobject.py>`_ (documentation planned).
    It has examples of calling into C code which is implemented as
    `a C extension library <https://github.com/numba/numba/blob/master/numba/_dictobject.c>`_.
    For a simpler example of calling into C library, see Bodo's I/O features like
    `get_file_size <https://github.com/IntelLabs/bodo/blob/master/bodo/io.py#L12>`_.
- | `Developer reference manual <http://numba.pydata.org/numba-doc/latest/developer/index.html>`_
    provides more details if necessary.

Bodo Development
----------------

Bodo implements Pandas and Numpy API as a DSL.
Data structures are implemented as Numba extensions, and
compiler stages are responsible for different levels of abstraction.
For example, `Series data type support <https://github.com/IntelLabs/bodo/blob/master/bodo/hiframes/pd_series_ext.py>`_
and `Series transformations <https://github.com/IntelLabs/bodo/blob/master/bodo/transforms/series_pass.py>`_
implement the `Pandas Series API <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html>`_.
Follow the pipeline for a simple function like `Series.sum()`
for initial understanding of the transformations.

Code Structures
---------------

Below is the high level structure of the code.

- `decorators.py` is the starting point, which defines decorators of Bodo.
  Currently just `@jit` is provided but more is expected.
- `compiler.py` defines the compiler pipeline for this decorator.
- `transforms` directory defines Bodo specific analysis and transformation passes.

  - `untyped_pass.py`: transforms the IR to remove features that Numba's type inference cannot support
    such as non-uniform dictionary input of `pd.DataFrame({})`.
  - `dataframe_pass.py`: converts data frame operations to Series and Array operations
    as much as possible to provide implementation and enable optimization.
    Creates specialized IR nodes for complex operations like Join.
  - `series_pass.py`: converts Series operations to array operations as much as possible
    to provide implementation and enable optimization.
  - `distributed_analysis.py`: analyzes the IR to decide parallelism of arrays and parfors
    for distributed transformation.
  - `distributed_pass.py`: parallelizes the IR for distributed execution and inserts MPI calls.

- `hiframes` directory provides Pandas functionality such as DataFrame, Series and Index.
- `ir` directory defines and implements Bodo specific IR nodes such as Sort and Join.
- `libs` directory provides supporting data structures and libraries such as strings,
  dictionary, quantiles, timsort. It also includes helper C extensions.
- `io` directory provides I/O support such as CSV, HDF5, Parquet and Numpy.
- `tests` provides unittests.

For each function implemented (either overloading Pandas or internal),
the following has to be specified:

- side effects for dead code elimination
- aliasing (inlining if necessary)
- array analysis
- distributed analysis (including array access analysis)
- distributed transformation

Building conda package::
  conda-build . -c defaults -c numba/label/dev -c conda-forge


Papers
------

These papers provide deeper dive in technical ideas
(may not be necessary for many developers):

- `Bodo paper on automatic parallelization for distributed memory <http://dl.acm.org/citation.cfm?id=3079099>`_
- `Bodo paper on system architecture versus Spark <http://dl.acm.org/citation.cfm?id=3103004>`_
- `Bodo Dataframe DSL approach <https://arxiv.org/abs/1704.02341>`_
- `ParallelAccelerator DSL approach <https://users.soe.ucsc.edu/~lkuper/papers/parallelaccelerator-ecoop17.pdf>`_
