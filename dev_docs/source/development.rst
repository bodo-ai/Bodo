.. _development:

Bodo Development
================

Getting Started
---------------

#. This `company slide deck <https://drive.google.com/open?id=1Vtbw-k9okgEc870Ad1wmKwUZQ0wJQEXc>`_ provides an overview of the company.
#. This `tech slide deck <https://drive.google.com/file/d/1mHrbjAEfP6p-o-bWJOVdtmKNEA7lreDt/view?usp=sharing>`_
   provides an overview of Bodo technology and software architecture.
#. Go over `a basic Pandas tutorial <https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html#min>`_.
#. Go over `a basic SQL tutorial <https://mode.com/sql-tutorial/introduction-to-sql>`_.
#. Read the rest of this page.
#. Install Bodo for development, see :ref:`install`.
#. Go over `getting started tutorial <https://github.com/Bodo-inc/Bodo/blob/master/tutorial/bodo_getting_started.ipynb>`_.
#. Go over `training tutorial <https://github.com/Bodo-inc/Bodo/blob/master/tutorial/bodo_tutorial.ipynb>`_.
#. Go over `Bodo user documentation <http://docs.bodo.ai/>`_.


Overview
--------

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


We use :code:`pytest` for testing. The tests are designed for up to
3 processors. Run the test suite on different
number of processors (should run in Bodo repo's main directory)::

    pytest -s -v -m "not slow" -W ignore
    mpiexec -n 2 pytest -s -v -m "not slow" -W ignore
    mpiexec -n 3 pytest -s -v -m "not slow" -W ignore


We have two customized `pytest markers <http://doc.pytest.org/en/latest/example/markers.html>`_:

1. :code:`slow` defined in `pytest.ini <https://github.com/Bodo-inc/Bodo/blob/master/pytest.ini>`_::
    
      pytest -s -v -m "slow" -W ignore
      pytest -s -v -m "not slow" -W ignore

   The :code:`not slow` flag skips some less necessary tests,
   which allows for faster testing. So it is used in the PR/merge pipeline.

   The nightly CI build&test pipeline runs the full test suite.
      
2. :code:`firsthalf` dynamically defined in `bodo/tests/conftest.py <https://github.com/Bodo-inc/Bodo/blob/master/bodo/tests/conftest.py>`_::

      pytest -s -v -m "firsthalf" -W ignore
      pytest -s -v -m "not firsthalf" -W ignore

   We use this marker in the nightly CI build&test pipeline due to limited memory available on azure.

Two markers can be used together::

   pytest -s -v -m "not slow and firsthalf" -W ignore

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

Currently our :code:`.flake8` config ignores a number of files, so whenever you are done working on a python file, run  `black <https://github.com/psf/black>`_, remove the file from :code:`.flake8`, and ensure `flake8 <http://flake8.pycqa.org/en/latest/>`_ does not raise any error.

We use the Google C++ code style guide
and enforce with `cpplint <https://github.com/cpplint/cpplint>`_.
We use `clang-format` as the formatter.
See `instructions in Pandas <https://pandas.pydata.org/pandas-docs/stable/development/contributing.html#c-cpplint>`_.


Code Coverage
---------------
We use `codecov <https://codecov.io/gh/Bodo-inc/Bodo>`_ for coverage reports. 
In `setup.cfg <https://github.com/Bodo-inc/Bodo/blob/package_config/setup.cfg>`_, there are two `coverage <https://coverage.readthedocs.io/en/coverage-5.0/>`_ configurations related sections.

To have a more accurate codecov report, during development, add :code:`# pragma: no cover` to numba compiled functions and dummy functions used for typing, which includes:

1. :code:`@numba.njit` functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/hiframes/pd_index_ext.py#L217>`_)
2. :code:`@numba.extending.register_jitable` functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/libs/int_arr_ext.py#L147>`_)
3. :code:`impl` (returned function) inside :code:`@overload` functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/libs/array_kernels.py#L636>`_)
4. :code:`impl` (returned function) inside :code:`@overload_method` functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/libs/str_arr_ext.py#L778>`_)
5. :code:`impl` (returned function) inside :code:`@numba.generated_jit` functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/hiframes/pd_dataframe_ext.py#L395>`_)
6. dummy functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/hiframes/pd_dataframe_ext.py#L1846>`_)


DevOps
----------

We currently have two build pipelines on `Azure DevOps <https://dev.azure.com/bodo-inc/Bodo/_build>`_:

1. Bodo-inc.Bodo: This pipeline is triggered whenever a pull request whose target branch is set to :code:`master` is created and following commits. This does not test on the full test suite in order to save time. A `codecov <https://codecov.io/gh/Bodo-inc/Bodo>`_ code coverage report is generated and uploaded for testing on Linux with one processor.

2. Bodo-build-binary: This pipeline is used for release and automatic nightly testing on full test suite, triggered by pushing tags. It has two stages. The first stage removes docstrings, builds the bodo binary and makes the artifact(:code:`bodo-inc.zip`) available for downloads. The second stage runs the full test suite with the binary we just built on Linux with 1, 2, and 3 processors. It is structured this way so that in case of emergency bug fix release, we can still download the binary without waiting for the tests to finish. 

The default :code:`TRIAL_PERIOD` is 14(days) set through Azure's UI, and this enviroment variable can be changed before manually triggering the build. 

:code:`MAX_CORE_COUNT` does not have a default value, it can be set through Azure's UI when manually triggering it.


Papers
------

These papers provide deeper dive in technical ideas
(may not be necessary for many developers):

- `Bodo paper on automatic parallelization for distributed memory <http://dl.acm.org/citation.cfm?id=3079099>`_
- `Bodo paper on system architecture versus Spark <http://dl.acm.org/citation.cfm?id=3103004>`_
- `Bodo Dataframe DSL approach <https://arxiv.org/abs/1704.02341>`_
- `ParallelAccelerator DSL approach <https://users.soe.ucsc.edu/~lkuper/papers/parallelaccelerator-ecoop17.pdf>`_
