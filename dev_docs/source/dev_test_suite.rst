.. _dev_test_suite:

Test Suite
----------

pytest
~~~~~~~
We use :code:`pytest` for testing. The tests are designed for up to
3 processors. Run the test suite on different
number of processors (should run in Bodo repo's main directory)::

    pytest -s -v -m "not slow" -W ignore
    mpiexec -n 2 pytest -s -v -m "not slow" -W ignore
    mpiexec -n 3 pytest -s -v -m "not slow" -W ignore


Example of running a specific test in bodo/test/test_file.py::

    pytest -s -v -m "not slow" -W ignore bodo/tests/test_date.py::test_datetime_operations



pytest markers
================
We have three customized `pytest markers <http://doc.pytest.org/en/latest/example/markers.html>`_:

1. :code:`slow` defined in `pytest.ini <https://github.com/Bodo-inc/Bodo/blob/master/pytest.ini>`_::
    
      pytest -s -v -m "slow" -W ignore
      pytest -s -v -m "not slow" -W ignore

   The :code:`not slow` flag skips some less necessary tests,
   which allows for faster testing. So it is used in the PR/merge pipeline.

   The nightly CI build&test pipeline runs the full test suite.
   Therefore, when new tests are added, if the tests take considerable amount of time and there are other tests for similar functionalities, it should be marked as slow. 
      
2. :code:`firsthalf` dynamically defined in `bodo/tests/conftest.py <https://github.com/Bodo-inc/Bodo/blob/master/bodo/tests/conftest.py>`_::

      pytest -s -v -m "firsthalf" -W ignore
      pytest -s -v -m "not firsthalf" -W ignore

   We use this marker in the nightly CI build&test pipeline due to limited memory available on azure.

3. :code:`s3` defined in `pytest.ini <https://github.com/Bodo-inc/Bodo/blob/master/pytest.ini>`_::

      pytest -s -v -m "s3" -W ignore

   This marker marks the tests that test for s3 file system. These tests will be skipped, if `boto3
   <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>`_ or `botocore
   <https://botocore.amazonaws.com/v1/documentation/api/latest/index.html>`_ is not installed.

More than one markers can be used together::
    
   pytest -s -v -m "not slow and firsthalf" -W ignore



pytest fixture
================

The purpose of test fixtures is to provide a fixed baseline upon which tests 
can reliably and repeatedly execute.
For example, Pytest fixture can be used when multiple tests use same input datas::
    
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
=====================
Pytest.mark. parameterize can also be used to test multiple inputs for a specific function::

    @pytest.mark.parametrize(
        "S",
        [
            pd.Series([True, False, False, True, True]),
            pd.Series([True, False, False, np.nan, True]),
        ],
    )
    def test_series_astype_bool_arr(S):
        # TODO: int, Int

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In some cases, we do not want to perfrom distributed testing. In such cases, we can use non-Bodo testing functions. 
List of Non-Bodo testing functions that can also be used while testing are

    1. assert
    2. pandas.testing.assert_series_equal
    3. pandas.testing.assert_frame_equal
    4. numpy.testing.assert_array_equal



Error Checking
~~~~~~~~~~~~~~~~~~~~
When the implementation of function does not fully encounter various types of possible input data, 
Numba starts to compare the given data type to other types to find right action for the given input.
If not found or all existing signatures failed, Numba falls back to object mode (eg. string type will be converted to unicode type). This potentially makes the program slow
and most importantly, the error message that Numba generates is not user friendly as it throws out pages of errors.
To prevent it and to provide users useful and meaningful message, we perform error checking. 
Depending on situations, we check for input data types and even their values.
We raise ``BodoError``, a subclass of python ``BaseException``, when the input is of wrong types or unsupported/invalid values.
Implementing ``BodoError`` from ``BaseExecption`` class instead of ``Exception`` was necessary because Numba sometimes catches ``Exeception`` and perform tasks accordingly instead of
just terminating the program. BodoError will terminate the program and provide simple error message for the users. 
Following is an example of our error checking for unsupported input::

    @overload_method(SeriesStrMethodType, "get")
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




Once error checking is implemented on a function, we should test whether the error checking is functional::

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
