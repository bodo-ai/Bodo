.. _development:

Bodo Development
================

Technology Overview and Architecture
------------------------------------

This `slide deck <https://drive.google.com/file/d/1mHrbjAEfP6p-o-bWJOVdtmKNEA7lreDt/view?usp=sharing>`_
provides an overview of Bodo technology and software architecture.


Bodo Development
----------------

Bodo implements Pandas and Numpy APIs as an embedded DSL.
Data structures are implemented as Numba extensions, and
compiler stages are responsible for different levels of abstraction.
For example, `Series data type support <https://github.com/IntelLabs/bodo/blob/master/bodo/hiframes/pd_series_ext.py>`_
and `Series transformations <https://github.com/IntelLabs/bodo/blob/master/bodo/transforms/series_pass.py>`_
implement the `Pandas Series API <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html>`_.
Follow the compiler pipeline (input/output IRs) for a simple function like
`Series.sum()` for initial understanding of the transformations.
See the `Numba development page <https://github.com/Bodo-inc/Bodo/blob/master/docs/development/numba.rst>`_
for information about Numba, which is critical for Bodo development.
See the `Bodo install page <https://github.com/Bodo-inc/Bodo/blob/master/docs/development/numba.rst>`_
for information about setting up the enviroment for Bodo development.

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


Debugging
---------
- `pdb <https://docs.python.org/3/library/pdb.html>`_: :code:`import pdb; pdb.set_trace()` for breakpoints

- `NUMBA_DEBUG_PRINT_AFTER <https://numba.pydata.org/numba-doc/dev/reference/envvars.html?highlight=numba_debug_print#envvar-NUMBA_DEBUG_PRINT_AFTER>`_ enviroment variable: 
  ::
      # example of printing after parfor pass
      export NUMBA_DEBUG_PRINT_AFTER='parfor_pass'
      # other common ones: 'bodo_distributed_pass', 'bodo_series_pass'

- mpiexec redirect stdout from differet processes to different files:
  ::
      export PYTHONUNBUFFERED=1 # set the enviroment variable 
      mpiexec -outfile-pattern="out_%r.log" -n 8 python small_test01.py

  or :
  ::
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
