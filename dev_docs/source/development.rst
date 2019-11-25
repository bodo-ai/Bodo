.. _development:

Bodo Development
================

Technology Overview and Architecture
------------------------------------

This `slide deck <https://drive.google.com/file/d/1mHrbjAEfP6p-o-bWJOVdtmKNEA7lreDt/view?usp=sharing>`_
provides an overview of Bodo technology and software architecture.

Bodo implements Pandas and Numpy APIs as an embedded DSL.
Data structures are implemented as Numba extensions, and
compiler stages are responsible for transforming different
levels of abstraction, optimization, and parallelization.
For example, `Series data type support <https://github.com/IntelLabs/bodo/blob/master/bodo/hiframes/pd_series_ext.py>`_
and `Series transformations <https://github.com/IntelLabs/bodo/blob/master/bodo/transforms/series_pass.py>`_
implement the `Pandas Series API <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html>`_.


Compiler Stages
---------------

`BodoCompiler <https://github.com/Bodo-inc/Bodo/blob/master/bodo/compiler.py#L68>`_
class defines the compiler pipeline. Below are the main stages.

- `TranslateByteCode`, ... before `BodoUntypedPass`:
  Numba's frontend passes that process function byte code, generate
  the IR, and prepare for type inference.
- `BodoUntypedPass`: transforms the IR to remove features that Numba's type
  inference cannot support such as non-uniform dictionary input of
  `pd.DataFrame({})`.
- `NopythonTypeInference`: Numba's type inference.
- `BodoDataFramePass`: converts data frame operations to Series and Array
  operations as much as possible to provide implementation and enable
  optimization. Creates specialized IR nodes for complex operations like Join.
- `BodoSeriesPass`: converts Series operations to array operations as much as
  possible to provide implementation and enable optimization.
- `ParforPass`: converts Numpy operations into parfors, fuses all parfors
  if possible, and performs basic optimizations such as copy propagation and
  dead code elimination.
- `BodoDistributedPass`: analyzes the IR to decide parallelism of arrays and
  parfors for distributed transformation, then
  parallelizes the IR for distributed execution and inserts MPI calls.
- `NoPythonBackend`: Numba's backend to generate LLVM IR and eventually binary.


For demonstration of these passes, follow the compiler pipeline (input/output IRs) for a simple function like
`Series.sum()` for initial understanding of the transformations.
See the `Numba development page <https://github.com/Bodo-inc/Bodo/blob/master/docs/development/numba.rst>`_
for information about Numba, which is critical for Bodo development.
See the `Bodo install page <https://github.com/Bodo-inc/Bodo/blob/master/docs/development/numba.rst>`_
for information about setting up the enviroment for Bodo development.


Sentinel Functions
------------------

Bodo transforms Pandas APIs (and others if needed) into *sentinel*
functions that can be analyzed and optimized throughout the pipeline.
Different stages of the compiler handle these functions if necessary,
with all the analysis for them available if needed.

For example, `get_series_data` function is used for getting the underlying
data array of a Series object. BodoSeriesPass removes this function
if the data array is available at that point in the program
(Series object was created using `init_series` and not altered).


For the pipline to handle a sentinel function properly
the following has to be specified:

- side effects for dead code elimination
- aliasing
- inlining (if necessary)
- array analysis
- distributed analysis (including array access analysis)
- distributed transformation

For example, `get_series_data` does not have side effects and can be removed
if output is not live. In addition, the output is aliased with the input,
and both have the same parallel distribution.


IR Extensions
-------------

Bodo uses IR extensions for operations that are too complex for
sentinel functions to represent. For example, Join and Aggregate nodes
represent `merge` and `groupby/aggregate` operations of Pandas respectively.
IR extensions have full transformation and analysis support (usually
more extensive that sentinel functions).


Code Structure
--------------

Below is the high level structure of the code.

- `decorators.py` is the starting point, which defines decorators of Bodo.
  Currently just `@jit` is provided but more is expected.
- `compiler.py` defines the compiler pipeline for this decorator.
- `transforms` directory defines Bodo specific analysis and transformation
  passes.
- `hiframes` directory provides Pandas functionality such as DataFrame,
  Series and Index.
- `ir` directory defines and implements Bodo specific IR nodes such as
  Sort and Join.
- `libs` directory provides supporting data structures and libraries such as
  strings, dictionary, quantiles, timsort. It also includes helper C
  extensions.
- `io` directory provides I/O support such as CSV, HDF5, Parquet and Numpy.
- `tests` provides unittests.


Test Suite
----------


We use `pytest` for testing and run the test suite on different
number of processors (should run in Bodo repo's main directory)::

    pytest -s -v -m "not slow" -W ignore
    mpiexec -n 2 pytest -s -v -m "not slow" -W ignore
    mpiexec -n 3 pytest -s -v -m "not slow" -W ignore


Debugging
---------
- `pdb <https://docs.python.org/3/library/pdb.html>`_: :code:`import pdb; pdb.set_trace()` for breakpoints

- `NUMBA_DEBUG_PRINT_AFTER <https://numba.pydata.org/numba-doc/dev/reference/envvars.html?highlight=numba_debug_print#envvar-NUMBA_DEBUG_PRINT_AFTER>`_
  enviroment variable::

    # example of printing after parfor pass
    export NUMBA_DEBUG_PRINT_AFTER='parfor_pass'
    # other common ones: 'bodo_distributed_pass', 'bodo_series_pass'

- mpiexec redirect stdout from differet processes to different files::

    export PYTHONUNBUFFERED=1 # set the enviroment variable 
    mpiexec -outfile-pattern="out_%r.log" -n 8 python small_test01.py

  or::

    # use the flag instead of setting the enviroment variable
    mpiexec -outfile-pattern="out_%r.log" -n 8 python -u small_test01.py


Code Style
----------

Bodo uses the PEP8 standard for Python code style.
We use `black <https://github.com/psf/black>`_ as formatter
and check format with `flake8 <http://flake8.pycqa.org/en/latest/>`_.

We use the Google C++ code style guide
and enforce with `cpplint <https://github.com/cpplint/cpplint>`_.
We use `clang-format` as the formatter.
See `instructions in Pandas <https://pandas.pydata.org/pandas-docs/stable/development/contributing.html#c-cpplint>`_.


Papers
------

These papers provide deeper dive in technical ideas
(may not be necessary for many developers):

- `Bodo paper on automatic parallelization for distributed memory <http://dl.acm.org/citation.cfm?id=3079099>`_
- `Bodo paper on system architecture versus Spark <http://dl.acm.org/citation.cfm?id=3103004>`_
- `Bodo Dataframe DSL approach <https://arxiv.org/abs/1704.02341>`_
- `ParallelAccelerator DSL approach <https://users.soe.ucsc.edu/~lkuper/papers/parallelaccelerator-ecoop17.pdf>`_
