# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests I/O error checking for CSV, Parquet, HDF5, etc.
"""
# TODO: Move error checking tests from test_io to here.

import os
import re

import numba
import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.utils.testing import ensure_clean
from bodo.utils.typing import BodoError, BodoWarning


def test_unsupported_error_checking(memory_leak_check):
    """make sure BodoError is raised for unsupported call"""
    # test an example I/O call
    def test_impl():
        return pd.read_spss("data.dat")

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_impl)()


# TODO: Add memory_leak_check when bugs are resolved.
def test_csv_invalid_path():
    """test error raise when CSV file path is invalid and the data types are provided
    explicitly (so that path check is done in C++ runtime).
    """

    def test_impl(fname):
        return pd.read_csv(fname, names=["A"], dtype={"A": np.int64})

    with pytest.raises(RuntimeError, match="invalid path"):
        bodo.jit(test_impl)("f.csv")


def test_csv_invalid_path_const(memory_leak_check):
    """test error raise when CSV file path provided as constant but is invalid."""

    def test_impl():
        return pd.read_csv("in_csv.csv")

    with pytest.raises(BodoError, match="No such file or directory"):
        bodo.jit(test_impl)()


def test_csv_repeat_args(memory_leak_check):
    """
    test error raise when an untyped pass function provides an argument
    as both an arg and a kwarg
    """

    def test_impl():
        return pd.read_csv("csv_data1.csv", filepath_or_buffer="csv_data1.csv")

    with pytest.raises(
        BodoError,
        match=r"pd.read_csv\(\) got multiple values for argument 'filepath_or_buffer'",
    ):
        bodo.jit(test_impl)()


def test_read_csv_incorrect_s3_credentials(memory_leak_check):
    """test error raise when AWS credentials are incorrect for csv
    file path passed by another bodo.jit function"""

    filename = "s3://test-pq-2/item.pq"
    # Save default developer mode value
    default_mode = numba.core.config.DEVELOPER_MODE
    # Test as a user
    numba.core.config.DEVELOPER_MODE = 0

    @bodo.jit
    def read(filename):
        df = pd.read_csv(filename)
        return df

    # Test with passing filename from bodo to bodo call error and S3
    def test_impl_csv(filename):
        return read(filename)

    with pytest.raises(BodoError, match="No response body"):
        bodo.jit(test_impl_csv)(filename)

    # Reset developer mode
    numba.core.config.DEVELOPER_MODE = default_mode


def test_io_error_nested_calls(memory_leak_check):
    """Test with passing incorrect filename from bodo to bodo call
    with local file"""
    # Save default developer mode value
    default_mode = numba.core.config.DEVELOPER_MODE
    # Test as a user
    numba.core.config.DEVELOPER_MODE = 0

    @bodo.jit
    def test_csv(filename):
        df = pd.read_csv(filename)
        return df

    def test_impl_csv(filename):
        return test_csv(filename)

    filename = "I_dont_exist.csv"
    with pytest.raises(BodoError, match="No such file or directory"):
        bodo.jit(test_impl_csv)(filename)

    @bodo.jit
    def test_pq(filename):
        df = pd.read_parquet(filename)
        return df

    def test_impl_pq(filename):
        return test_pq(filename)

    filename = "I_dont_exist.pq"
    with pytest.raises(BodoError, match="error from pyarrow: FileNotFoundError"):
        bodo.jit(test_impl_pq)(filename)

    # Reset developer mode
    numba.core.config.DEVELOPER_MODE = default_mode


# TODO(Nick): Add a test for Parallel version with both offset and count.
@pytest.mark.slow
def test_np_io_sep_unsupported(datapath, memory_leak_check):
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, sep=" ")
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    with pytest.raises(
        BodoError, match=r"np.fromfile\(\): sep argument is not supported"
    ):
        # Test that we cannot swap the value of sep.
        bodo_func()


@pytest.mark.slow
def test_csv_infer_type_error(datapath):
    ints = [0] * 1000
    strings = ["a"] * 1000
    df = pd.DataFrame({"A": ints + strings})
    filepath = datapath("test_csv_infer_type_error.csv", check_exists=False)
    with ensure_clean(filepath):
        if bodo.get_rank() == 0:
            df.to_csv(filepath, index=False)
        bodo.barrier()
        message = r"pd.read_csv\(\): Bodo could not infer dtypes correctly."
        with pytest.raises(TypeError, match=message):
            bodo.jit(lambda: pd.read_csv(filepath), all_returns_distributed=True)()
        with pytest.raises(TypeError, match=message):
            bodo.jit(lambda: pd.read_csv(filepath), distributed=False)()


def test_pseudo_exception(datapath, memory_leak_check):
    """Test removal of ForceLiteralArg message"""

    def test_csv(fname):
        df = pd.read_csv(fname)
        return df.clip()

    fname = datapath("csv_data1.csv")

    with pytest.raises(Exception) as excinfo:
        bodo.jit(test_csv)(fname)
    # Save full traceback as a string
    csv_track = excinfo.getrepr(style="native")
    assert "Pseudo-exception" not in str(csv_track)

    def test_pq(file_name):
        df = pd.read_parquet(file_name)
        return df.A.WRONG()

    fname = datapath("groupby3.pq")
    with pytest.raises(Exception) as excinfo:
        bodo.jit(test_pq)(fname)

    pq_track = excinfo.getrepr(style="native")
    assert "Pseudo-exception" not in str(pq_track)


@pytest.mark.slow
def test_csv_chunksize_type(memory_leak_check):
    """
    Test read_csv(): 'chunksize' with a wrong value or type
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl1(fname):
        return pd.read_csv(fname, chunksize=-1)

    def impl2(fname):
        return pd.read_csv(fname, chunksize="no thanks")

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) 'chunksize' must be a constant integer >= 1 if provided.",
    ):
        bodo.jit(impl1)(fname)
    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) 'chunksize' must be a constant integer >= 1 if provided.",
    ):
        bodo.jit(impl2)(fname)


@pytest.mark.slow
def test_csv_nrows_type():
    """
    Test read_csv(): 'nrows' wrong value or type

    TODO: re-add memory_leak_check, see BE-1545
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl1():
        return pd.read_csv(fname, nrows=-2)

    def impl2():
        return pd.read_csv(fname, nrows="wrong")

    with pytest.raises(ValueError, match="integer >= 0"):
        bodo.jit(impl1)()
    with pytest.raises(BodoError, match="must be an integer"):
        bodo.jit(impl2)()


@pytest.mark.slow
def test_csv_skiprows_type(memory_leak_check):
    """
    Test read_csv(): 'skiprows' wrong value or type
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl1():
        return pd.read_csv(fname, skiprows=-1)

    def impl2():
        return pd.read_csv(fname, skiprows="wrong")

    def impl3():
        return pd.read_csv(
            fname, skiprows=lambda x: x > 1, names=["X", "Y", "Z", "MM", "AA"]
        )

    with pytest.raises(BodoError, match="integer >= 0"):
        bodo.jit(impl1)()
    with pytest.raises(
        BodoError, match="'skiprows' must be an integer or list of integers."
    ):
        bodo.jit(impl2)()
    with pytest.raises(BodoError, match="callable not supported yet"):
        bodo.jit(impl3)()


def test_to_csv_compression_kwd_arg():
    """checks that the appropriate errors are raised when compression argument to to_csv"""
    from bodo.tests.test_io import check_CSV_write

    @bodo.jit
    def impl(df, f_name):
        return df.to_csv(f_name, compression="zip")

    def impl2(df, f_name):
        return df.to_csv(f_name, compression=None)

    def impl3(df, f_name):
        return df.to_csv(f_name)

    def non_compressed_read_impl(f_name):
        return pd.read_csv(f_name, compression=None)

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    check_CSV_write(
        impl2,
        df,
        pandas_filename="pandas_out.zip",
        bodo_filename="bodo_out.zip",
        read_impl=non_compressed_read_impl,
    )

    check_CSV_write(
        impl3,
        df,
        pandas_filename="pandas_out.csv",
        bodo_filename="bodo_out.zip",
        read_impl=non_compressed_read_impl,
    )

    with pytest.raises(
        BodoError,
        match=r".*DataFrame.to_csv\(\): 'compression' argument supports only None, which is the default in JIT code.*",
    ):
        impl(df, "bodo_out.zip")


def test_to_csv_unsupported_kwds_error():
    """checks that the unsupported keywords to to_csv raise an error"""
    msg1 = (
        r".*DataFrame.to_csv\(\): mode parameter only supports default value w[\s | .]*"
    )
    msg2 = r".*DataFrame.to_csv\(\): encoding parameter only supports default value None[\s | .]*"
    msg3 = r".*DataFrame.to_csv\(\): errors parameter only supports default value strict[\s | .]*"
    msg4 = r".*DataFrame.to_csv\(\): storage_options parameter only supports default value None[\s | .]*"

    @bodo.jit
    def impl1(df, f_name):
        return df.to_csv(f_name, mode="a")

    @bodo.jit
    def impl2(df, f_name):
        return df.to_csv(f_name, encoding="ascii")

    @bodo.jit
    def impl3(df, f_name):
        return df.to_csv(f_name, errors="replace")

    @bodo.jit
    def impl4(df, f_name):
        return df.to_csv(f_name, storage_options={"anon": True})

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})

    with pytest.raises(
        BodoError,
        match=msg1,
    ):
        impl1(df, "foo.csv")

    with pytest.raises(
        BodoError,
        match=msg2,
    ):
        impl2(df, "foo.csv")

    with pytest.raises(
        BodoError,
        match=msg3,
    ):
        impl3(df, "foo.csv")

    with pytest.raises(
        BodoError,
        match=msg4,
    ):
        impl4(df, "foo.csv")


def test_to_csv_filepath_warning():
    """We raise a best effort warning when the user tries to write to a constant filepath that would normally be compressed"""

    @bodo.jit
    def impl(df):
        return df.to_csv("foo.zip")

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    msg = r".*DataFrame.to_csv\(\): 'compression' argument defaults to None in JIT code, which is the only supported value.*"
    with pytest.warns(BodoWarning, match=msg):
        impl(df)


@pytest.mark.slow
def test_csv_skiprows_type_nonconstant(memory_leak_check):
    """
    Test read_csv(): 'skiprows' with the wrong type when its not a constant
    throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl(skiprow_list):
        return pd.read_csv(
            fname, skiprows=skiprow_list[0], names=["X", "Y", "Z", "MM", "AA"]
        )

    skiprow_list = ["abcd"]
    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\): 'skiprows' must be an integer or list of integers. Found type unicode_type.",
    ):
        bodo.jit(impl)(skiprow_list)


# TODO: [BE-1532] checking values at runtime doesn't pass memory_leak_check
@pytest.mark.slow
def test_csv_skiprows_nonconstant_incorrect_values():
    """
    Test read_csv(): 'skiprows' with the negative numbers when its not a constant
    throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl(skiprow_list):
        return pd.read_csv(
            fname, skiprows=skiprow_list[0], names=["X", "Y", "Z", "MM", "AA"]
        )

    skiprow_list = [-2]
    with pytest.raises(ValueError, match="integer >= 0"):
        bodo.jit(impl)(skiprow_list)

    @bodo.jit
    def f2():
        return [1, -3, 0]

    def impl2():
        return pd.read_csv(fname, skiprows=f2(), names=["X", "Y", "Z", "MM", "AA"])

    with pytest.raises(ValueError, match="integer >= 0"):
        bodo.jit(impl2)()


@pytest.mark.slow
def test_csv_skiprows_list_type(memory_leak_check):
    """
    Test read_csv(): 'skiprows' with list has wrong value or type
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl1():
        return pd.read_csv(fname, skiprows=[3, 2, 5, -1, -10])

    def impl2():
        return pd.read_csv(fname, skiprows=["wrong", "type"])

    with pytest.raises(BodoError, match="integer >= 0"):
        bodo.jit(impl1)()
    with pytest.raises(BodoError, match="must be an integer"):
        bodo.jit(impl2)()


@pytest.mark.slow
def test_csv_skiprows_list_type_nonconstant(memory_leak_check):
    """
    Test read_csv(): 'skiprows' list with the wrong type when its not a constant
    throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    @bodo.jit
    def f():
        return ["abcd", "efg"]

    def impl():
        return pd.read_csv(fname, skiprows=f(), names=["X", "Y", "Z", "MM", "AA"])

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\): 'skiprows' must be an integer or list of integers. Found type list\\(unicode_type\\)",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_nrows_type_nonconstant(memory_leak_check):
    """
    Test read_csv(): 'nrows' with the wrong type when its not a constant
    throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl(nrows_list):
        return pd.read_csv(fname, nrows=nrows_list[0])

    nrows_list = ["abcd"]
    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\): 'nrows' must be an integer. Found type unicode_type.",
    ):
        bodo.jit(impl)(nrows_list)


@pytest.mark.slow
def test_csv_sep_and_delimiter(memory_leak_check):
    """
    Test read_csv(): Providing both 'sep' and 'delimiter'
    throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        return pd.read_csv(fname, ",", delimiter=".")

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) Specified a 'sep' and a 'delimiter'\\; you can only specify one.",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_filename_type(memory_leak_check):
    """
    Test read_csv(): Test that passing a constant
    filename with the wrong type throws a reasonable error.
    """

    def impl():
        return pd.read_csv(1)

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) 'filepath_or_buffer' must be a string.",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_variable_filename_type(memory_leak_check):
    """
    Test read_csv(): Test that passing a non-constant
    filename with the wrong type throws a reasonable error.
    """

    def impl(fname_list):
        return pd.read_csv(fname_list[0], dtype={"A": str}, names=["A"])

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\): 'filepath_or_buffer' must be a string. Found type: int64",
    ):
        bodo.jit(impl)([1])


@pytest.mark.slow
def test_csv_multichar_sep(memory_leak_check):
    """
    Test read_csv(): Test that passing a multicharacter
    'sep' throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        return pd.read_csv(fname, sep="ab")

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) 'sep' is an invalid separator.",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_multichar_delimiter(memory_leak_check):
    """
    Test read_csv(): Test that passing a multicharacter
    'delimiter' throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        return pd.read_csv(fname, delimiter="ab")

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) 'delimiter' is an invalid separator.",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_unsupported_arg_kwarg(memory_leak_check):
    """
    Test read_csv(): Test that passing a repeated arg and
    kwarg for an unsupported arg throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        # squeeze is repeated
        return pd.read_csv(
            fname, ",", None, "infer", ["A", "B", "C"], None, None, False, squeeze=False
        )

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) got multiple values for argument 'squeeze'.",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_unknown_argument(memory_leak_check):
    """
    Test read_csv(): Test that passing an unknown
    argument throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        # column_names is not a valid argument
        return pd.read_csv(fname, column_names=["A", "B", "C"])

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) Unknown argument\\(s\\) \\['column_names'\\] provided.",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_unsupported_arg_mismatch(memory_leak_check):
    """
    Test read_csv(): Test that passing an unsupported arg that doesn't
    match the default throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        # squeeze is provided but not supported. It doesn't match
        # the default so it should raise an error.
        return pd.read_csv(
            fname,
            ",",
            None,
            "infer",
            ["A", "B", "C", "D", "E"],
            None,
            None,
            True,  # squeeze argument is here
        )

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) arguments \\['squeeze'\\] not supported yet",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_unsupported_kwarg_mismatch(memory_leak_check):
    """
    Test read_csv(): Test that passing an unsupported kwarg that doesn't
    match the default throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        # comment is provided but not supported. It doesn't match
        # the default so it should raise an error.
        return pd.read_csv(fname, comment=[2])

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) arguments \\['comment'\\] not supported yet",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_header_unsupported(memory_leak_check):
    """
    Test read_csv(): Test that passing an unsupported value for
    header throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        return pd.read_csv(fname, header="A")

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) 'header' should be one of 'infer', 0, or None",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_names_unsupported(memory_leak_check):
    """
    Test read_csv(): Test that passing an unsupported value for
    names throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        return pd.read_csv(fname, names="A")

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) 'names' should be a constant list if provided",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_index_col_unsupported(memory_leak_check):
    """
    Test read_csv(): Test that passing an unsupported value for
    index_col throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        return pd.read_csv(fname, index_col=True)

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) 'index_col' must be a constant integer, constant string that matches a column name, or False",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_usecols_unsupported(memory_leak_check):
    """
    Test read_csv(): Test that passing an unsupported value for
    usecols throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        return pd.read_csv(fname, usecols=True)

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) 'usecols' must be a constant list of columns names or column indices if provided",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_dtype_unsupported(memory_leak_check):
    """
    Test read_csv(): Test that passing an unsupported value for
    dtype throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        return pd.read_csv(fname, dtype=bodo.string_type)

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) 'dtype' does not support unicode_type",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_parse_dates_unsupported(memory_leak_check):
    """
    Test read_csv(): Test that passing an unsupported value for
    parse_dates throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        return pd.read_csv(fname, parse_dates=True)

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) 'parse_dates' must be a constant list of column names or column indices if provided",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_compression_unsupported(memory_leak_check):
    """
    Test read_csv(): Test that passing an unsupported value for
    compression throws a reasonable error.
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        return pd.read_csv(fname, compression="gz")

    with pytest.raises(
        BodoError,
        match="pd.read_csv\\(\\) 'compression' must be one of \\['infer', 'gzip', 'bz2', 'zip', 'xz', None\\]",
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_csv_escapechar_value():
    """
    Test read_csv(): 'escapechar' wrong value or type
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    # wrong datatype
    def impl1():
        return pd.read_csv(fname, escapechar=3)

    # > 1-char
    def impl2():
        return pd.read_csv(fname, escapechar="wrong")

    # same as sep
    def impl3():
        return pd.read_csv(fname, escapechar=",")

    # newline
    def impl4():
        return pd.read_csv(fname, escapechar="\n")

    with pytest.raises(BodoError, match="must be a one-character string"):
        bodo.jit(impl1)()
    with pytest.raises(BodoError, match="must be a one-character string"):
        bodo.jit(impl2)()
    with pytest.raises(BodoError, match="must not be equal to 'sep'"):
        bodo.jit(impl3)()
    with pytest.raises(BodoError, match="newline as 'escapechar' is not supported"):
        bodo.jit(impl4)()


@pytest.mark.slow
def test_csv_names_usecols_err():
    """
    Test read_csv(): names < usecols
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl1():
        return pd.read_csv(fname, usecols=[0, 2], names=["A"])

    with pytest.raises(
        BodoError, match="number of used columns exceeds the number of passed names"
    ):
        bodo.jit(impl1)()


@pytest.mark.slow
def test_csv_usecols_not_found():
    """
    Test read_csv(): usecols column not found
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl1():
        return pd.read_csv(fname, usecols=["B"], names=["A"])

    with pytest.raises(BodoError, match="does not match columns"):
        bodo.jit(impl1)()


@pytest.mark.slow
def test_csv_sample_nrows_size(memory_leak_check):
    """
    Test read_csv(): 'sample_nrows' with a wrong value or type
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    # negative values
    def impl1(fname):
        return pd.read_csv(fname, sample_nrows=-1)

    # string
    def impl2(fname):
        return pd.read_csv(fname, sample_nrows="no thanks")

    # nonconstant
    def impl3(fname, ilist):
        return pd.read_csv(fname, sample_nrows=ilist[0])

    with pytest.raises(
        BodoError,
        match=re.escape(
            "pd.read_csv() 'sample_nrows' must be a constant integer >= 1 if provided."
        ),
    ):
        bodo.jit(impl1)(fname)
    with pytest.raises(
        BodoError,
        match=re.escape(
            "pd.read_csv() 'sample_nrows' must be a constant integer >= 1 if provided."
        ),
    ):
        bodo.jit(impl2)(fname)
    ilist = [10]
    with pytest.raises(
        BodoError,
        match=" 'sample_nrows' argument as a constant int",
    ):
        bodo.jit(impl3)(fname, ilist)


@pytest.mark.slow
def test_json_sample_nrows_size(memory_leak_check):
    """
    Test read_json(): 'sample_nrows' with a wrong value or type
    """
    fname = os.path.join("bodo", "tests", "data", "example.json")

    # negative values
    def impl1(fname):
        return pd.read_json(fname, sample_nrows=-1)

    # string
    def impl2(fname):
        return pd.read_json(fname, sample_nrows="no thanks")

    # nonconstant
    def impl3(fname, ilist):
        return pd.read_json(fname, sample_nrows=ilist[0])

    with pytest.raises(
        BodoError,
        match=re.escape(
            "pd.read_json() 'sample_nrows' must be a constant integer >= 1 if provided."
        ),
    ):
        bodo.jit(impl1)(fname)
    with pytest.raises(
        BodoError,
        match=re.escape(
            "pd.read_json() 'sample_nrows' must be a constant integer >= 1 if provided."
        ),
    ):
        bodo.jit(impl2)(fname)
    ilist = [10]
    with pytest.raises(
        BodoError,
        match=" 'sample_nrows' argument as a constant int",
    ):
        bodo.jit(impl3)(fname, ilist)


# TODO: fix memory_leak_check
@pytest.mark.slow
def test_read_csv_sample_nrows_error(datapath):
    """Test read_csv with sample_nrows argument where data type change
    after sample_nrows
    See test_read_csv_sample_nrows for fix

    Args:
        datapath (fixture function): get path to test file
        memory_leak_check (fixture function): check memory leak in the test.

    """
    fname = datapath("large_data.csv")

    def impl1():
        return pd.read_csv(fname)

    # 1st 100 rows for column b is empty
    with pytest.raises(TypeError, match="Bodo could not infer dtypes correctly."):
        bodo.jit(impl1)()


# TODO: fix memory_leak_check
@pytest.mark.slow
def test_read_json_sample_nrows_error(datapath):
    """Test read_json with sample_nrows argument where data type change
    after sample_nrows
    See test_read_json_sample_nrows for fix

    Args:
        datapath (fixture function): get path to test file
        memory_leak_check (fixture function): check memory leak in the test.

    """
    # run only on 1 processor
    # TODO: [BE-3260]
    if bodo.get_size() != 1:
        return
    fname = datapath("large_data.json")

    def impl1():
        return pd.read_json(fname, lines=True, orient="records")

    # 1st 100 rows for column a is integer, next 100 are floats
    with pytest.raises(
        TypeError, match="cannot safely cast non-equivalent float64 to int64"
    ):
        bodo.jit(impl1)()
