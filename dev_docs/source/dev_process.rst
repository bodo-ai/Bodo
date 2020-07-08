Development Process
===================

IDE
---

Using a modern IDE that provides smart error catching like `VS Code <https://code.visualstudio.com/>`_
(which uses `pylint <https://www.pylint.org/>`_ by default) is highly recommended.

.. _dev_test_suite:

Test Suite
----------

pytest
~~~~~~~
We use :code:`pytest` for testing. The tests are designed for up to
3 processors. Run the test suite on different
number of processors (should run in Bodo repo's main directory)::

    pytest -s -v -m "not slow"
    mpiexec -n 2 pytest -s -v -m "not slow"
    mpiexec -n 3 pytest -s -v -m "not slow"


Example of running a specific test in bodo/test/test_file.py::

    pytest -s -v -m "not slow" bodo/tests/test_date.py::test_datetime_operations



pytest markers
^^^^^^^^^^^^^^

We have three customized `pytest markers <http://doc.pytest.org/en/latest/example/markers.html>`_:

1. :code:`slow` defined in `pytest.ini <https://github.com/Bodo-inc/Bodo/blob/master/pytest.ini>`_::
    
      pytest -s -v -m "slow"
      pytest -s -v -m "not slow"

   The :code:`not slow` flag skips some less necessary tests,
   which allows for faster testing. So it is used in the PR/merge pipeline.

   The nightly CI build & test pipeline runs the full test suite.
   Therefore, when new tests are added, if the tests take considerable amount of time
   and there are other tests for similar functionalities, it should be marked as slow.
      
2. :code:`firsthalf` dynamically defined in `bodo/tests/conftest.py <https://github.com/Bodo-inc/Bodo/blob/master/bodo/tests/conftest.py>`_::

      pytest -s -v -m "firsthalf"
      pytest -s -v -m "not firsthalf"

   We use this marker in the nightly CI build & test pipeline due to limited memory available on azure.

3. :code:`s3` defined in `pytest.ini <https://github.com/Bodo-inc/Bodo/blob/master/pytest.ini>`_::

      pytest -s -v -m "s3"

   This marker marks the tests that test for s3 file system. These tests will be skipped, if `boto3
   <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>`_ or `botocore
   <https://botocore.amazonaws.com/v1/documentation/api/latest/index.html>`_ is not installed.

4. :code:`hdfs` defined in `pytest.ini <https://github.com/Bodo-inc/Bodo/blob/master/pytest.ini>`_::

      pytest -s -v -m "hdfs"

  This marker marks the tests that test for hdfs file system.
  These tests will be skipped, if `hdfs3 <https://hdfs3.readthedocs.io/en/latest/>`_ is not installed.

More than one markers can be used together::
    
   pytest -s -v -m "not slow and firsthalf"



pytest fixture
^^^^^^^^^^^^^^

The purpose of test fixtures is to provide a fixed baseline upon which tests 
can reliably and repeatedly execute.
For example, Pytest fixture can be used when multiple tests use same inputs::
    
    @pytest.fixture(params=[pd.Series(["New_York", "Lisbon", "Tokyo", "Paris", "Munich"]),
                            pd.Series(["1234", "ABCDF", "ASDDDD", "!@#@@", "FAFFD"])])
    def test_sr(request):
        return request.param

    def test_center_width_noint(test_sr):
    """
    tests error for center with the argument 'width' being non-integer type
    """

        def impl(test_sr):
            return test_sr.str.center(width="1", fillchar="*")

        with pytest.raises(BodoError, match="expected an int object"):
            bodo.jit(impl)(test_sr)



pytest parameterize
^^^^^^^^^^^^^^^^^^^

Pytest.mark.parameterize can also be used to test multiple inputs for a specific function::

    @pytest.mark.parametrize(
        "S",
        [
            pd.Series([True, False, False, True, True]),
            pd.Series([True, False, False, np.nan, True]),
        ],
    )
    def test_series_astype_bool_arr(S):

        def test_impl(S):
            return S.astype("float32")

        check_func(test_impl, (S,))



Bodo Testing Function
~~~~~~~~~~~~~~~~~~~~~~
Bodo uses a function called ``check_func`` to validate the result of Bodo function against that of Pandas.
Following code is an example of using `check_func`::

    def test_series_dt64_timestamp_cmp():
        """Test Series.dt comparison with pandas.timestamp scalar
        """
        def test_impl(S, t):
            return S == t

        S = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
        timestamp = pd.to_datetime("2018-04-24")

        # compare series(dt64) with a timestamp and a string
        check_func(test_impl, (S, timestamp))

`check_func` performs 3 testings. 
    - Sequential testing
    - distributed testing with all the processors having the same size of data
    - distributed testing with processors having different sizes of data. 
        - The second last processor will have 1 less element
        - The last processor will have 1 more element
        - Must provide large enough size of data (at least input length of 5) to make sure
          that none of the processor end up with not having any input data. 

Each test is independent from one another, so during development/debugging, individual tests can be commented out.
In certain cases, distributed tests are not performed. Check the comments in `check_func <https://github.com/Bodo-inc/Bodo/blob/master/bodo/tests/utils.py>`_


Other useful testing functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases, we do not want to perfrom distributed testing. In such cases, we can use non-Bodo testing functions. 
List of Non-Bodo testing functions that can also be used while testing are

    1. assert
    2. pandas.testing.assert_series_equal
    3. pandas.testing.assert_frame_equal
    4. numpy.testing.assert_array_equal



Error Checking
~~~~~~~~~~~~~~
When the implementation of a function (e.g. overload) does not support a particular input set (i.e. argument types),
Numba may throw hard-to-understand generic and long errors that are not user friendly.
To prevent this and to provide useful and meaningful error messages to users,
we need to provide comprehensive error checking to all supported APIs.

Depending on the situation, we check for input data types and even their literal values if necessary.
We raise ``BodoError``, a subclass of python ``BaseException``, when the input is of wrong types or unsupported/invalid values.
Implementing ``BodoError`` from ``BaseExecption`` class instead of ``Exception`` is currently necessary because Numba sometimes
catches ``Exeception`` and tries to continue with its own error path instead of
just terminating compilation. BodoError will terminate compilation and provide simple error message for the user.
Following is an example of our error checking for unsupported input::

    @overload_method(SeriesStrMethodType, "get", no_unliteral=True)
    def overload_str_method_get(S_str, i):
        arr_typ = S_str.stype.data
        if (
            arr_typ != string_array_split_view_type
            and arr_typ != list_string_array_type
            and arr_typ != string_array_type
        ):
            raise BodoError(
                "Series.str.get(): only supports input type of Series(list(str)) "
                "and Series(str)"
            )


Once error checking is implemented for a function, we should test whether the error checking is functional::

    @pytest.mark.parametrize(
        "input",
        [
            pd.Series([1, 2, 3]),
            # pd.Series([(1, 2, 3), (3, 4, 5)])  # TODO: support unboxing Series of tuples
        ],
    )
    def test_get_input(input):
        """
        tests error for get with the input series not being ListStringArrayType or
        StringArrayType
        """

        def impl(input):
            return input.str.get(1)

        with pytest.raises(BodoError, match="only supports input type of"):
            bodo.jit(impl)(input)


.. _dev_code_structure:

Code Structure
--------------

Below is the high level structure of the code.

- ``decorators.py`` is the starting point, which defines decorators of Bodo.
  Currently just ``@jit`` is provided but more is expected.
- ``compiler.py`` defines the compiler pipeline for this decorator.
- ``transforms`` directory defines Bodo specific analysis and transformation
  passes.
- ``hiframes`` directory provides Pandas functionality such as DataFrame,
  Series and Index.
- ``ir`` directory defines and implements Bodo specific IR nodes such as
  Sort and Join.
- ``libs`` directory provides supporting data structures and libraries such as
  strings, dictionary, quantiles, timsort. It also includes helper C
  extensions.
- ``io`` directory provides I/O support such as CSV, HDF5, Parquet and Numpy.
- ``tests`` provides unittests.

.. _dev_debugging:

Debugging
---------

Debugging the Python code
~~~~~~~~~~~~~~~~~~~~~~~~~~
- `pdb <https://docs.python.org/3/library/pdb.html>`_: setting breakpoints
  using :code:`import pdb; pdb.set_trace()` and inspecting variables is key
  for debugging Bodo's python code such as overloads and transformations.

- Debugging overloads: Numba's overload handling may hide errors and raise unrelated
  and misleading exceptions instead. One can debug these cases by setting a
  breakpoint right before the return of the relevant overload, and stepping through
  Numba's internal code until the actual error is raised.

- `NUMBA_DEBUG_PRINT_AFTER <https://numba.pydata.org/numba-doc/dev/reference/envvars.html?highlight=numba_debug_print#envvar-NUMBA_DEBUG_PRINT_AFTER>`_
  enviroment variable prints the IR after specified compiler passes,
  which helps debugging transformations significantly::

      # example of printing after parfor pass
      export NUMBA_DEBUG_PRINT_AFTER='parfor_pass'

  Other common one: ``'bodo_distributed_pass', 'bodo_series_pass'``, ``bodo_dataframe_pass``
- mpiexec redirect stdout from differet processes to different files::

    export PYTHONUNBUFFERED=1  # avoid IO buffering
    mpiexec -outfile-pattern="out_%r.log" -n 8 python small_test01.py

  or::

    # use "-u" instead of setting the enviroment variable
    mpiexec -outfile-pattern="out_%r.log" -n 8 python -u small_test01.py


Debugging the C++ code
~~~~~~~~~~~~~~~~~~~~~~~~~~

Sanitizers of GCC can be used for debugging C/C++ code.
The compilation option need to be added to `setup.py`::

    eca = ["-std=c++11", "-fsanitize=address"]
    ela = ["-std=c++11", "-fsanitize=address"]

In the docker, the next step is to add the library::

    export LD_PRELOAD=/root/miniconda3/envs/BODODEV/lib/libasan.so.5

Then we can see running times error using sanitizers.

.. _dev_codestyle:

Code Style
----------

Bodo uses the PEP8 standard for Python code style.
We use `black <https://github.com/psf/black>`_ as formatter
and check format with `flake8 <http://flake8.pycqa.org/en/latest/>`_.

Currently our :code:`.flake8` config ignores a number of files,
so whenever you are done working on a python file, run  `black <https://github.com/psf/black>`_,
remove the file from :code:`.flake8`, and ensure `flake8 <http://flake8.pycqa.org/en/latest/>`_ does not raise any error.

We use the Google C++ code style guide
and enforce with `cpplint <https://github.com/cpplint/cpplint>`_.
We use `clang-format` as the formatter.
See `instructions in Pandas <https://pandas.pydata.org/pandas-docs/stable/development/contributing.html#c-cpplint>`_.

Removing Unused Imports
~~~~~~~~~~~~~~~~~~~~~~~
`autoflake` can be used to remove unused imports across all the files in the repository.

First install `autoflake`::

    conda install autoflake -c conda-forge

Following command remove unused import in a file::

    autoflake --in-place --remove-all-unused-imports <filename>

`-r` flag can be added to the above command to apply `autoflake` to all the files in a directory. 
More information can be found `here <https://github.com/myint/autoflake>`_.

.. _dev_codecoverage:

Code Coverage
-------------

We use `codecov <https://codecov.io/gh/Bodo-inc/Bodo>`_ for coverage reports. 
In `setup.cfg <https://github.com/Bodo-inc/Bodo/blob/package_config/setup.cfg>`_,
there are two `coverage <https://coverage.readthedocs.io/en/coverage-5.0/>`_ configurations related sections.

To have a more accurate codecov report during development, add :code:`# pragma: no cover`
to numba compiled functions and dummy functions used for typing, which includes:

1. :code:`@numba.njit` functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/hiframes/pd_index_ext.py#L217>`_)
2. :code:`@numba.extending.register_jitable` functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/libs/int_arr_ext.py#L147>`_)
3. :code:`impl` (returned function) inside :code:`@overload` functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/libs/array_kernels.py#L636>`_)
4. :code:`impl` (returned function) inside :code:`@overload_method` functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/libs/str_arr_ext.py#L778>`_)
5. :code:`impl` (returned function) inside :code:`@numba.generated_jit` functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/hiframes/pd_dataframe_ext.py#L395>`_)
6. dummy functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/hiframes/pd_dataframe_ext.py#L1846>`_)

.. _dev_devops:

DevOps
------

We currently have three build pipelines on `Azure DevOps <https://dev.azure.com/bodo-inc/Bodo/_build>`_:

1. Bodo-inc.Bodo: This pipeline is triggered whenever a pull request whose target branch is set
   to :code:`master` is created, as well as its later commits. This does not test on the full
   test suite in order to save time. A `codecov <https://codecov.io/gh/Bodo-inc/Bodo>`_ code coverage
   report is generated and uploaded after testing on Linux with one processor.

2. Bodo-build-binary: This pipeline is used for automatic nightly testing on full test suite.
   It can also be triggered by pushing tags. It has two stages. The first stage removes docstrings,
   builds the bodo binary and makes the artifact(:code:`bodo-inc.zip`) available for downloads.
   The second stage runs the full test suite with the binary we just built on Linux with 1, 2, and 3 processors.
   It is structured this way so that in case of emergency bug fix release,
   we can still download the binary without waiting for the tests to finish.

3. Bodo-build-binary-obfuscated: This pipeline is used for release and automatic nightly testing
   on full test suite, triggered by pushing tags. This pipeline is performing exactly the same operations
   as :code:`Bodo-build-binary` pipeline does, except that the files in the artifact are obfuscated.
   We use this to build binaries for customers.

For the two release pipelines (Bodo-build-binary and Bodo-build-binary-obfuscated), there are some variables used,
and they can all be changed manually triggering the pipelines:

- :code:`CHECK_LICENSE_EXPIRED` has a default value of 1 set through Azure's UI. If set to 1, binary will do license check of expiration date
- :code:`CHECK_LICENSE_CORE_COUNT` has a default value of 1 set through Azure's UI. If set to 1, binary will do license check of max core count

:code:`OBFUSCATE` is set to 0 for :code:`Bodo-build-binary` pipeline and 1 for :code:`Bodo-build-binary-obfuscated` pipeline.

.. _dev_benchmark:

Performance Benchmarking
------------------------

We use AWS EC2 instances for performance benchmark on Bodo.
This is essentially to check the performance variations based on commits to master branch.
Similar to our nightly build, benchmarking is set to run regularly. 
To set up this infrastructure there are 3 things that should be constructed. 

    1. AWS EC2 instance
    2. AWS CodePipeline
    3. AWS CloudFormat

CodePipeline performs 4 tasks.

    1. Download source code from github
    2. Build the source on AWS build server. Build script for AWS build server can be found `here <https://github.com/Bodo-inc/Bodo/blob/master/buildspec.yml>`_
    3. Deploy the build artifact to EC2 instance
    4. Run whatever the user provides with `scripts <https://github.com/Bodo-inc/Bodo/blob/master/appspec.yml>`_
        - Run Bodo Benchmarking
        - Run TPCH Benchmarking
        - Upload the result to Bodo/`Benchmark_log repository <https://github.com/Bodo-inc/benchmark_logs>`_

CloudFormat performs 3 tasks.

    1. It will turn on the EC2 instance based on the schedule we set to reduce the cost.
    2. After turning on EC2 instance, CloudFormat will also trigger the pipeline.
    3. Turn off the EC2 instance based on the schedule. Make sure to give enough time to allow the pipeline to finish its tasks.
