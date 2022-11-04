# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests I/O for CSV, HDF5, etc."""
import io
import os
import subprocess
import unittest

import h5py
import numba
import numpy as np
import pandas as pd
import pytest
from numba.core.ir_utils import build_definitions, find_callname

import bodo
from bodo.pandas_compat import pandas_version
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    DeadcodeTestPipeline,
    _get_dist_arg,
    _test_equal_guard,
    check_func,
    count_array_REPs,
    count_parfor_REPs,
    get_rank,
    get_start_end,
    reduce_sum,
)
from bodo.utils.testing import ensure_clean
from bodo.utils.typing import BodoError
from bodo.utils.utils import is_call_assign


def compress_file(fname, dummy_extension=""):
    assert not os.path.isdir(fname)
    if bodo.get_rank() == 0:
        subprocess.run(["gzip", "-k", "-f", fname])
        subprocess.run(["bzip2", "-k", "-f", fname])
        if dummy_extension != "":
            os.rename(fname + ".gz", fname + ".gz" + dummy_extension)
            os.rename(fname + ".bz2", fname + ".bz2" + dummy_extension)
    bodo.barrier()
    return [fname + ".gz" + dummy_extension, fname + ".bz2" + dummy_extension]


def remove_files(file_names):
    if bodo.get_rank() == 0:
        for fname in file_names:
            os.remove(fname)
    bodo.barrier()


def compress_dir(dir_name):
    if bodo.get_rank() == 0:
        for fname in [
            f
            for f in os.listdir(dir_name)
            if f.endswith(".csv") and os.path.getsize(dir_name + "/" + f) > 0
        ]:
            subprocess.run(["gzip", "-f", fname], cwd=dir_name)
    bodo.barrier()


def uncompress_dir(dir_name):
    if bodo.get_rank() == 0:
        for fname in [f for f in os.listdir(dir_name) if f.endswith(".gz")]:
            subprocess.run(["gunzip", fname], cwd=dir_name)
    bodo.barrier()


# TODO: Implement Delta Lake in a way similar to Iceberg:
# https://bodo.atlassian.net/browse/BE-3048
@pytest.mark.skip
def test_read_parquet_from_deltalake(memory_leak_check):
    def impl():
        return pd.read_parquet("bodo/tests/data/example_deltalake")

    py_output = pd.DataFrame({"value": [1, 1, 2, 3, 2, 3]})
    check_func(impl, (), py_output=py_output, check_dtype=False)


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


def test_bodo_upcast(datapath):
    """
    Tests the _bodo_upcast_to_float64 custom argument to read csv. This ensures that a file
    which would otherwise cause typing issues will upcast to a shared type when possible
    """
    # Create strings of ints and floats
    ints = ["0"] * 1000
    floats = ["1.1"] * 1000
    df = pd.DataFrame({"A": ints + floats})
    filepath = datapath("test_mixed_int_float.csv", check_exists=False)
    with ensure_clean(filepath):
        if bodo.get_rank() == 0:
            df.to_csv(filepath, index=False)
        bodo.barrier()
        message = r"pd.read_csv\(\): Bodo could not infer dtypes correctly."
        with pytest.raises(TypeError, match=message):
            bodo.jit(lambda: pd.read_csv(filepath), all_returns_distributed=True)()
        bodo.jit(
            lambda: pd.read_csv(filepath, _bodo_upcast_to_float64=True),
            distributed=False,
        )()


def test_to_csv_none_arg0(memory_leak_check):
    """checks that passing None as the filepath argument is properly supported"""

    def impl(df):
        return df.to_csv(path_or_buf=None)

    def impl2(df):
        return df.to_csv()

    def impl3(df):
        return df.to_csv(None)

    df = pd.DataFrame({"A": np.arange(100)})

    check_func(impl, (df,), only_seq=True)
    check_func(impl2, (df,), only_seq=True)
    check_func(impl3, (df,), only_seq=True)

    # for the distributed cases, the output for each rank is the same as calling
    # df.to_csv(None) on the distributed dataframe
    py_out_1d = _get_dist_arg(df).to_csv(None)
    check_func(impl, (df,), only_1D=True, py_output=py_out_1d)
    check_func(impl2, (df,), only_1D=True, py_output=py_out_1d)
    check_func(impl3, (df,), only_1D=True, py_output=py_out_1d)

    py_out_1d_vars = _get_dist_arg(df, var_length=True).to_csv(None)
    check_func(impl, (df,), only_1DVar=True, py_output=py_out_1d_vars)
    check_func(impl2, (df,), only_1DVar=True, py_output=py_out_1d_vars)
    check_func(impl3, (df,), only_1DVar=True, py_output=py_out_1d_vars)


def test_to_csv_filepath_as_kwd_arg(memory_leak_check):
    """checks that passing the filepath as a keyword argument is properly supported"""

    def impl(df, f_name):
        return df.to_csv(path_or_buf=f_name)

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    check_CSV_write(impl, df)


def test_basic_paralel_write(memory_leak_check):
    """does a basic test of to_csv with no arguments"""

    def impl(df, f_name):
        return df.to_csv(f_name)

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    check_CSV_write(impl, df)


def test_to_csv_sep_kwd_arg(memory_leak_check):
    """tests the sep keyword argument to to_csv."""

    def impl_none(df):
        return df.to_csv(None, sep="-")

    def impl(df, f_name):
        return df.to_csv(f_name, sep=":")

    df = pd.DataFrame({"A": np.arange(100)})

    check_to_csv_string_output(df, impl_none)
    check_CSV_write(impl, df)


def test_to_csv_na_rep_kwd_arg(memory_leak_check):
    """tests the na_rep keyword argument to to_csv."""

    def impl_none(df):
        return df.to_csv(None, na_rep="NA_VALUE")

    def impl(df, f_name):
        return df.to_csv(f_name, na_rep="-1")

    df = pd.DataFrame({"A": np.arange(10)}, dtype="Int64")
    df["A"][0] = None

    check_to_csv_string_output(df, impl_none)
    check_CSV_write(impl, df)


def test_to_csv_float_format_kwd_arg(memory_leak_check):
    """tests the float_format keyword argument to to_csv."""

    def impl_none(df):
        return df.to_csv(None, float_format="%.3f")

    def impl(df, f_name):
        return df.to_csv(f_name, float_format="%.3f")

    # This should generate floats with a large number of decimal places
    df = pd.DataFrame({"A": [x / 7 for x in range(10)]})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(impl, df)


def test_to_csv_columns_kwd_arg(memory_leak_check):
    """tests the columns keyword argument to to_csv. List input is not currently supported, see BE-1505"""

    def impl_none(df):
        return df.to_csv(None, columns=("A", "C"))

    def impl(df, f_name):
        return df.to_csv(f_name, columns=("A", "C"))

    def impl_list(df, f_name):
        return df.to_csv(f_name, columns=["A", "C"])

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(impl, df)
    check_CSV_write(impl_list, df)


# Header argument tested in test_csv_header_write_read


def test_to_csv_index_kwd_arg(memory_leak_check):
    """tests the index keyword argument to to_csv"""

    def impl_none(df):
        return df.to_csv(None, index=False)

    def impl(df, f_name):
        return df.to_csv(f_name, index=False)

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})

    check_to_csv_string_output(df, impl_none)
    check_CSV_write(impl, df)


def test_to_csv_index_label_kwd_arg(memory_leak_check):
    """tests the index_label keyword argument to to_csv"""

    def impl_none(df):
        return df.to_csv(None, index_label=False)

    def impl(df, f_name):
        return df.to_csv(f_name, index_label=False)

    def impl_none2(df):
        return df.to_csv(None, index_label="LABEL_")

    def impl2(df, f_name):
        return df.to_csv(f_name, index_label="LABEL_")

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    check_to_csv_string_output(df, impl_none)
    check_to_csv_string_output(df, impl_none2)
    check_CSV_write(impl, df)
    check_CSV_write(impl2, df)


def test_to_csv_quoting_kwd_arg(memory_leak_check):
    """tests the quoting keyword argument to to_csv"""
    import csv

    def impl_none(df):
        return df.to_csv(None, quoting=csv.QUOTE_ALL)

    def impl(df, f_name):
        return df.to_csv(f_name, quoting=csv.QUOTE_ALL)

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(
        impl,
        df,
    )


def test_to_csv_quotechar_kwd_arg(memory_leak_check):
    """tests the quotechar keyword argument to to_csv"""
    import csv

    def impl_none(df):
        return df.to_csv(None, quoting=csv.QUOTE_ALL, quotechar="Q")

    def impl(df, f_name):
        return df.to_csv(f_name, quoting=csv.QUOTE_ALL, quotechar="X")

    def read_impl(f_name):
        return pd.read_csv(f_name, quotechar="X")

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(
        impl,
        df,
        read_impl=read_impl,
    )


def test_to_csv_line_terminator_kwd_arg(memory_leak_check):
    """tests the line_terminator keyword argument to to_csv"""

    def impl_none(df):
        return df.to_csv(None, line_terminator="__LINE_TERMINATION__")

    def impl(df, f_name):
        return df.to_csv(f_name, line_terminator="\t")

    def read_impl(f_name):
        return pd.read_csv(f_name, lineterminator="\t")

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(
        impl,
        df,
        read_impl=read_impl,
    )


def test_to_csv_chunksize_kwd_arg(memory_leak_check):
    """tests the chunksize keyword argument to to_csv"""

    def impl_none(df):
        return df.to_csv(None, chunksize=7)

    def impl(df, f_name):
        return df.to_csv(f_name, chunksize=7)

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(
        impl,
        df,
    )


@pytest.mark.slow
def test_to_csv_date_format_kwd_arg(memory_leak_check):
    """tests the date_format keyword argument to to_csv."""

    def impl_none(df):
        return df.to_csv(None, date_format="%a, %b, %Y, %Z, %x")

    def impl(df, f_name):
        return df.to_csv(f_name, date_format="%W, %z, %f, %S, %x")

    df = pd.DataFrame(
        {"A": pd.date_range(start="1998-04-24", end="2000-04-29", periods=100)}
    )
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(
        impl,
        df,
    )


def test_to_csv_doublequote_escapechar_kwd_args(memory_leak_check):
    """tests the doublequote and escapechar keyword argument to to_csv.
    Doublequote and escapechar need to be tested together, as escapechar is
    the char used to escape when not double quoting.
    """

    def impl_none(df):
        return df.to_csv(
            None, doublequote=False, quotechar="a", sep="-", escapechar="E"
        )

    def impl(df, f_name):
        return df.to_csv(
            f_name, doublequote=False, quotechar="a", sep="-", escapechar="E"
        )

    def read_impl(f_name):
        return pd.read_csv(
            f_name, doublequote=False, quotechar="a", sep="-", escapechar="E"
        )

    df = pd.DataFrame({"A": ["a - a - a - a"] * 10})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(impl, df, read_impl=read_impl)


@pytest.mark.slow
def test_to_csv_decimal_kwd_arg(memory_leak_check):
    """tests the decimal keyword argument to to_csv"""

    def impl_none(df):
        return df.to_csv(None, decimal="_")

    def impl(df, f_name):
        return df.to_csv(f_name, decimal="_")

    def read_impl(f_name):
        return pd.read_csv(f_name, decimal="_")

    # This should generate floats with a large number of decimal places
    df = pd.DataFrame({"A": [x / 7 for x in range(10)]})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(impl, df, read_impl=read_impl)


@pytest.mark.slow
def test_read_csv_bad_dtype_column(datapath, memory_leak_check):
    """Checks calling read_csv() with columns in the dtype that
    aren't in the DataFrame. This raises a warning so the code
    should still execute."""

    fname = datapath("csv_data_infer1.csv")

    def test_impl(fname):
        dtype = {"B": "float64", "I_AM_A_MISSING_COLUMN": pd.Int32Dtype()}
        return pd.read_csv(fname, dtype=dtype)

    # Set check_dtype=False for nullable differences
    check_func(test_impl, (fname,), check_dtype=False)


def test_read_csv_nonascii(datapath, memory_leak_check):
    """Checks calling read_csv() with non-ascii data"""

    fname = datapath("csv_data_nonascii1.csv")

    def test_impl(fname):
        return pd.read_csv(fname)

    check_func(test_impl, (fname,))


@pytest.mark.slow
def test_csv_remove_col0_used_for_len(datapath, memory_leak_check):
    """read_csv() handling code uses the first column for creating RangeIndex of the
    output dataframe. In cases where the first column array is dead, it should be
    replaced by an alternative live array. This test makes sure this replacement happens
    properly.
    """
    fname = datapath("csv_data1.csv")
    fname_gzipped = fname + ".gz"

    def impl():
        df = pd.read_csv(fname, names=["A", "B", "C", "D"], compression=None)
        return df.C

    def impl2():
        df = pd.read_csv(fname_gzipped, names=["A", "B", "C", "D"], compression="gzip")
        return df.C

    check_func(impl, (), only_seq=True)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit()(impl)()
        check_logger_msg(stream, "Columns loaded ['C']")

    if bodo.get_rank() == 0:
        subprocess.run(["gzip", "-k", "-f", fname])
    bodo.barrier()
    with ensure_clean(fname_gzipped):
        check_func(impl2, (), only_seq=True)
        with set_logging_stream(logger, 1):
            bodo.jit()(impl2)()
            check_logger_msg(stream, "Columns loaded ['C']")


@pytest.mark.slow
def test_csv_remove_col_keep_date(datapath, memory_leak_check):
    """Test parse_date position matches usecols after removing unused column
    See [BE-2561]"""
    fname = datapath("csv_data_date1.csv")

    def test_impl(fname):
        df = pd.read_csv(
            fname,
            names=["A", "B", "C", "D"],
            dtype={"A": int, "B": float, "C": str, "D": int},
            parse_dates=["C"],
        )
        return df["C"]

    check_func(test_impl, (fname,))


@pytest.mark.slow
def test_csv_usecols_parse_dates(datapath, memory_leak_check):
    """Test usecols with parse_date See [BE-2544]"""
    fname = datapath("csv_data_date1.csv")

    def test_impl(fname):
        df = pd.read_csv(
            fname,
            names=["A", "B", "C", "D"],
            dtype={"A": int, "B": float, "C": str, "D": int},
            parse_dates=["C"],
            usecols=["A", "C", "D"],
        )
        return df

    check_func(test_impl, (fname,))


@pytest.mark.slow
def test_csv_usecols_names_args(datapath, memory_leak_check):
    """Test usecols and names argument together"""
    fname = datapath("example.csv")

    # subset for both names and usecols
    def impl1(fname):
        df = pd.read_csv(fname, names=["A", "B"], usecols=[0, 2])
        return df

    check_func(impl1, (fname,))

    # all column names and subset for usecols
    def impl2(fname):
        df = pd.read_csv(fname, names=["A", "B", "C", "D", "E"], usecols=[0, 2, 3])
        return df

    check_func(impl2, (fname,))

    # colnames > usecols but usecols has duplicates
    def impl3(fname):
        df = pd.read_csv(fname, names=["A", "B", "C"], usecols=[0, 2, 1, 0, 1])
        return df

    check_func(impl3, (fname,))

    # few names + dtypes=None + usecols=None
    def impl4(fname):
        df = pd.read_csv(fname, names=["A", "B", "C"])
        return df

    # Ignore index check See [BE-2596]
    check_func(impl4, (fname,), reset_index=True)

    # few names + dtypes + usecols=None
    # def impl5(fname):
    #     df = pd.read_csv(
    #         fname, names=["A", "B", "C"], dtypes={"A": "float", "B": "str", "C": "bool"}
    #     )
    #     return df

    # check_func(impl5, (fname,), reset_index=True)

    # colnames > usecols
    def impl6(fname):
        df = pd.read_csv(fname, names=["A", "B", "C", "D", "E"], usecols=[0])
        return df

    check_func(impl6, (fname,))


@pytest.mark.slow
def test_h5_remove_dead(datapath, memory_leak_check):
    """make sure dead hdf5 read calls are removed properly"""
    fname = datapath("lr.hdf5")

    def impl():
        f = h5py.File(fname, "r")
        X = f["points"][:, :]
        f.close()

    bodo_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(impl)
    bodo_func()
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    fir._definitions = build_definitions(fir.blocks)
    for stmt in fir.blocks[0].body:
        assert not (
            is_call_assign(stmt)
            and find_callname(fir, stmt.value) == ("h5read", "bodo.io.h5_api")
        )


def test_csv_bool1(datapath, memory_leak_check):
    """Test boolean data in CSV files.
    Also test extra separator at the end of the file
    which requires index_col=False.
    """

    def test_impl(fname):
        dtype = {"A": "int", "B": "bool", "C": "float"}
        return pd.read_csv(
            fname, names=list(dtype.keys()), dtype=dtype, index_col=False
        )

    # passing file name as argument to exercise value-based dispatch
    fname = datapath("csv_data_bool1.csv")
    check_func(test_impl, (fname,))

    compressed_names = compress_file(fname)
    try:
        for fname in compressed_names:
            check_func(test_impl, (fname,))
    finally:
        remove_files(compressed_names)


def test_csv_int_na1(datapath, memory_leak_check):
    fname = datapath("csv_data_int_na1.csv")

    def test_impl(fname):
        dtype = {"A": "int", "B": "Int32"}
        return pd.read_csv(
            fname, names=list(dtype.keys()), dtype=dtype, compression="infer"
        )

    check_func(test_impl, (fname,))

    compressed_names = compress_file(fname)
    try:
        for f in compressed_names:
            check_func(test_impl, (f,))
    finally:
        remove_files(compressed_names)

    # test reading csv file with non ".csv" extension
    new_fname = fname[:-4] + ".custom"  # change .csv to .custom
    if bodo.get_rank() == 0:
        os.rename(fname, new_fname)
    bodo.barrier()
    try:
        check_func(test_impl, (new_fname,))
    finally:
        if bodo.get_rank() == 0 and os.path.exists(new_fname):
            os.rename(new_fname, fname)
        bodo.barrier()


def test_csv_int_na2(datapath, memory_leak_check):
    fname = datapath("csv_data_int_na1.csv")

    def test_impl(fname, compression):
        dtype = {"A": "int", "B": pd.Int32Dtype()}
        return pd.read_csv(
            fname, names=list(dtype.keys()), dtype=dtype, compression=compression
        )

    check_func(test_impl, (fname, "infer"))

    compressed_names = compress_file(fname, dummy_extension=".dummy")
    try:
        check_func(test_impl, (compressed_names[0], "gzip"))
        check_func(test_impl, (compressed_names[1], "bz2"))
    finally:
        remove_files(compressed_names)


def test_csv_bool_na(datapath, memory_leak_check):
    fname = datapath("bool_nulls.csv")

    def test_impl(fname):
        # TODO: support column 1 which is bool with NAs when possible with
        # Pandas dtypes
        # see Pandas GH20591
        dtype = {"ind": "int32", "B": "bool"}
        return pd.read_csv(fname, names=list(dtype.keys()), dtype=dtype, usecols=[0, 2])

    check_func(test_impl, (fname,))

    compressed_names = compress_file(fname)
    try:
        for fname in compressed_names:
            check_func(test_impl, (fname,))
    finally:
        remove_files(compressed_names)


def test_csv_fname_comp(datapath, memory_leak_check):
    """Test CSV read with filename computed across Bodo functions"""

    @bodo.jit
    def test_impl(data_folder):
        return load_func(data_folder)

    @bodo.jit
    def load_func(data_folder):
        fname = data_folder + "/csv_data1.csv"
        return pd.read_csv(fname, header=None)

    data_folder = os.path.join("bodo", "tests", "data")
    # should not raise exception
    test_impl(data_folder)


def test_write_csv_parallel_unicode(memory_leak_check):
    def test_impl(df, fname):
        df.to_csv(fname)

    bodo_func = bodo.jit(all_args_distributed_block=True)(test_impl)
    S1 = ["¡Y tú quién te crees?", "🐍⚡", "大处着眼，小处着手。"] * 2
    S2 = ["abc¡Y tú quién te crees?", "dd2🐍⚡", "22 大处着眼，小处着手。"] * 2
    df = pd.DataFrame({"A": S1, "B": S2})
    hp_fname = "test_write_csv1_bodo_par_unicode.csv"
    pd_fname = "test_write_csv1_pd_par_unicode.csv"
    with ensure_clean(pd_fname), ensure_clean(hp_fname):
        start, end = get_start_end(len(df))
        bdf = df.iloc[start:end]
        bodo_func(bdf, hp_fname)
        bodo.barrier()
        if get_rank() == 0:
            test_impl(df, pd_fname)
            pd.testing.assert_frame_equal(
                pd.read_csv(hp_fname), pd.read_csv(pd_fname), check_column_type=False
            )


@pytest.mark.smoke
def test_h5_read_seq(datapath, memory_leak_check):
    def test_impl(fname):
        f = h5py.File(fname, "r")
        X = f["points"][:]
        f.close()
        return X

    # passing function name as value to test value-based dispatch
    fname = datapath("lr.hdf5")
    check_func(test_impl, (fname,), only_seq=True)


def test_h5_read_const_infer_seq(datapath, memory_leak_check):
    fname = datapath("")

    def test_impl():
        p = fname + "lr"
        f = h5py.File(p + ".hdf5", "r")
        s = "po"
        X = f[s + "ints"][:]
        f.close()
        return X

    check_func(test_impl, (), only_seq=True)


def test_h5_read_parallel(datapath, memory_leak_check):
    fname = datapath("lr.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        X = f["points"][:]
        Y = f["responses"][:]
        f.close()
        return X.sum() + Y.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl(), decimal=2)
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


@pytest.mark.skip(
    "H5py bug breaks boolean arrays, https://github.com/h5py/h5py/issues/1847"
)
def test_h5_filter(datapath, memory_leak_check):
    fname = datapath("h5_test_filter.h5")

    def test_impl():
        f = h5py.File(fname, "r")
        b = np.arange(11) % 3 == 0
        X = f["test"][b, :, :, :]
        f.close()
        return X

    bodo_func = bodo.jit(distributed=["X"])(test_impl)
    n = 4  # len(test_impl())
    start, end = get_start_end(n)
    np.testing.assert_allclose(bodo_func(), test_impl()[start:end])


def test_h5_slice1(datapath, memory_leak_check):
    fname = datapath("h5_test_filter.h5")

    def test_impl():
        f = h5py.File(fname, "r")
        X = f["test"][:, 1:, :, :]
        f.close()
        return X

    bodo_func = bodo.jit(distributed=["X"])(test_impl)
    n = 11  # len(test_impl())
    start, end = get_start_end(n)
    np.testing.assert_allclose(bodo_func(), test_impl()[start:end])


def test_h5_slice2(datapath, memory_leak_check):
    fname = datapath("lr.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        X = f["points"][:, 1]
        f.close()
        return X

    bodo_func = bodo.jit(distributed=["X"])(test_impl)
    n = 101  # len(test_impl())
    start, end = get_start_end(n)
    np.testing.assert_allclose(bodo_func(), test_impl()[start:end])


def test_h5_read_group(datapath, memory_leak_check):
    fname = datapath("test_group_read.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        g1 = f["G"]
        X = g1["data"][:]
        f.close()
        return X.sum()

    bodo_func = bodo.jit(test_impl)
    assert bodo_func() == test_impl()


def test_h5_file_keys(datapath, memory_leak_check):
    fname = datapath("test_group_read.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        s = 0
        for gname in f.keys():
            X = f[gname]["data"][:]
            s += X.sum()
        f.close()
        return s

    bodo_func = bodo.jit(test_impl, h5_types={"X": bodo.int64[:]})
    assert bodo_func() == test_impl()
    # test using locals for typing
    bodo_func = bodo.jit(test_impl, locals={"X": bodo.int64[:]})
    assert bodo_func() == test_impl()


def test_h5_group_keys(datapath, memory_leak_check):
    fname = datapath("test_group_read.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        g1 = f["G"]
        s = 0
        for dname in g1.keys():
            X = g1[dname][:]
            s += X.sum()
        f.close()
        return s

    bodo_func = bodo.jit(test_impl, h5_types={"X": bodo.int64[:]})
    assert bodo_func() == test_impl()


@pytest.mark.smoke
def test_h5_write(memory_leak_check):
    # run only on 1 processor
    if bodo.get_size() != 1:
        return

    def test_impl(A, fname):
        f = h5py.File(fname, "w")
        dset1 = f.create_dataset("A", A.shape, "f8")
        dset1[:] = A
        f.close()

    fname = "test_w.hdf5"
    n = 11
    A = np.arange(n).astype(np.float64)
    with ensure_clean(fname):
        bodo.jit(
            test_impl, returns_maybe_distributed=False, args_maybe_distributed=False
        )(A, fname)
        f = h5py.File(fname, "r")
        A2 = f["A"][:]
        f.close()
        np.testing.assert_array_equal(A, A2)


def test_h5_group_write(memory_leak_check):
    # run only on 1 processor
    if bodo.get_size() != 1:
        return

    def test_impl(A, fname):
        f = h5py.File(fname, "w")
        g1 = f.create_group("AA")
        g2 = g1.create_group("BB")
        dset1 = g2.create_dataset("A", A.shape, "f8")
        dset1[:] = A
        f.close()

    fname = "test_w.hdf5"
    n = 11
    A = np.arange(n).astype(np.float64)
    with ensure_clean(fname):
        bodo.jit(test_impl)(A, fname)
        f = h5py.File(fname, "r")
        A2 = f["AA"]["BB"]["A"][:]
        f.close()
        np.testing.assert_array_equal(A, A2)


@pytest.mark.smoke
def test_np_io1(datapath, memory_leak_check):
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64)
        return A

    bodo_func = bodo.jit(test_impl, returns_maybe_distributed=False)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


def test_np_io2(datapath, memory_leak_check):
    fname = datapath("np_file1.dat")
    # parallel version
    def test_impl():
        A = np.fromfile(fname, np.float64)
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


def test_np_io3(memory_leak_check):
    def test_impl(A):
        if get_rank() == 0:
            A.tofile("np_file_3.dat")

    bodo_func = bodo.jit(test_impl)
    n = 111
    np.random.seed(0)
    A = np.random.ranf(n)
    with ensure_clean("np_file_3.dat"):
        bodo_func(A)
        if get_rank() == 0:
            B = np.fromfile("np_file_3.dat", np.float64)
            np.testing.assert_almost_equal(A, B)


def test_np_io4(memory_leak_check):
    # parallel version
    def test_impl(n):
        A = np.arange(n)
        A.tofile("np_file_4.dat")

    bodo_func = bodo.jit(test_impl)
    n1 = 111000
    n2 = 111
    A1 = np.arange(n1)
    A2 = np.arange(n2)
    with ensure_clean("np_file_4.dat"):
        bodo_func(n1)
        B1 = np.fromfile("np_file_4.dat", np.int64)
        np.testing.assert_almost_equal(A1, B1)

        bodo.barrier()
        bodo_func(n2)
        B2 = np.fromfile("np_file_4.dat", np.int64)
        np.testing.assert_almost_equal(A2, B2)


def test_np_io5(datapath, memory_leak_check):
    # Test count optional argument small
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, count=10)
        return A

    check_func(test_impl, (), only_seq=True)


def test_np_io6(datapath, memory_leak_check):
    # Test count optional argument huge
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, count=100000000)
        return A

    check_func(test_impl, (), only_seq=True)


def test_np_io7(datapath, memory_leak_check):
    # Test offset optional argument
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, offset=10)
        return A

    check_func(test_impl, (), only_seq=True)


def test_np_io8(datapath, memory_leak_check):
    # Test offset and count optional arguments
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, offset=10, count=10)
        return A

    check_func(test_impl, (), only_seq=True)


def test_np_io9(datapath, memory_leak_check):
    # Test count optional argument small, parallel
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, count=10)
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


def test_np_io10(datapath, memory_leak_check):
    # Test count optional argument huge, parallel
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, count=100000000)
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


def test_np_io11(datapath, memory_leak_check):
    # Test offset optional argument, parallel
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, offset=10)
        print(A.shape, A)
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    v1 = bodo_func()
    v2 = test_impl()
    np.testing.assert_almost_equal(v1, v2)


def test_np_io12(datapath, memory_leak_check):
    # Test offset and count optional arguments, parallel
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, offset=10, count=10)
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


@pytest.mark.slow
def test_np_io13(datapath, memory_leak_check):
    # Test fromfile with all keyword arguments
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(file=fname, dtype=np.float64, count=10, sep="", offset=10)
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


@pytest.mark.slow
def test_np_io14(datapath, memory_leak_check):
    # Test fromfile with all positional arguments
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, 10, "", 10)
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


def test_csv_double_box(datapath, memory_leak_check):
    """Make sure boxing the output of read_csv() twice doesn't cause crashes
    See dataframe boxing function for extra incref of native arrays.
    """
    fname = datapath("csv_data1.csv")

    def test_impl():
        df = pd.read_csv(fname, header=None)
        print(df)
        return df

    bodo_func = bodo.jit(test_impl)
    print(bodo_func())


def test_csv_header_none(datapath, memory_leak_check):
    """Test header=None in read_csv() when column names are not provided, so numbers
    should be assigned as column names.
    """
    fname = datapath("csv_data1.csv")

    def test_impl():
        return pd.read_csv(fname, header=None)

    bodo_func = bodo.jit(returns_maybe_distributed=False)(test_impl)
    b_df = bodo_func()
    p_df = test_impl()
    # convert column names from integer to string since Bodo only supports string names
    p_df.columns = [str(c) for c in p_df.columns]
    pd.testing.assert_frame_equal(
        b_df, p_df, check_dtype=False, check_column_type=False
    )


def test_csv_sep_arg(datapath, memory_leak_check):
    """Test passing 'sep' argument as JIT argument in read_csv()"""
    fname = datapath("csv_data2.csv")

    def test_impl(fname, sep):
        return pd.read_csv(fname, sep=sep)

    check_func(
        test_impl,
        (
            fname,
            "|",
        ),
        check_dtype=False,
    )
    # testing reading whole lines with sep="\n"
    # This is no long supported in pandas 1.4
    if pandas_version == (1, 3):
        check_func(
            test_impl,
            (
                fname,
                "\n",
            ),
            check_dtype=False,
        )
    else:
        assert pandas_version == (1, 4), "Check if this test is still valid"
        with pytest.raises(
            BodoError, match=r".*Specified \\n as separator or delimiter.*"
        ):
            bodo.jit(test_impl)(fname, "\n")

    compressed_names = compress_file(fname)
    try:
        for fname in compressed_names:
            check_func(
                test_impl,
                (
                    fname,
                    "|",
                ),
                check_dtype=False,
            )
    finally:
        remove_files(compressed_names)


def test_csv_int_none(datapath, memory_leak_check):
    """Make sure int columns that have nulls later in the data (not seen by our 100 row
    type inference step) are handled properly
    """

    def test_impl(fname):
        df = pd.read_csv(fname)
        return df

    fname = datapath("csv_data_int_none.csv")
    check_func(test_impl, (fname,), check_dtype=False)


def test_csv_dtype_col_ind(datapath, memory_leak_check):
    """Make sure integer column index works for referring to columns in 'dtype'"""

    def test_impl(fname):
        df = pd.read_csv(fname, dtype={3: str})
        return df

    fname = datapath("csv_data_infer1.csv")
    check_func(test_impl, (fname,))


def test_csv_usecols_names(datapath, memory_leak_check):
    """test passing column names in usecols"""

    def test_impl(fname):
        df = pd.read_csv(fname, usecols=["A", "B"])
        return df

    fname = datapath("csv_data_infer1.csv")
    check_func(test_impl, (fname,), check_dtype=False)


def test_csv_sep_whitespace(datapath, memory_leak_check):
    """test that using all whitespace"""

    def test_impl(fname):
        df = pd.read_csv(fname, sep=r"\s+")
        return df

    fname = datapath("csv_data_infer1.csv")
    check_func(test_impl, (fname,), check_dtype=False)


def test_csv_spark_header(datapath, memory_leak_check):
    """Test reading Spark csv outputs containing header & infer dtypes"""
    fname1 = datapath("example_single.csv")
    fname2 = datapath("example_multi.csv")

    def test_impl(fname):
        return pd.read_csv(fname)

    py_output = pd.read_csv(datapath("example.csv"))
    check_func(test_impl, (fname1,), py_output=py_output)
    check_func(test_impl, (fname2,), py_output=py_output)

    for fname in (fname1, fname2):
        compress_dir(fname)
        try:
            check_func(test_impl, (fname,), py_output=py_output)
        finally:
            uncompress_dir(fname)

    # test reading a directory of csv files not ending in ".csv" extension
    dirname = fname2
    if bodo.get_rank() == 0:
        # rename all .csv files in directory to .custom
        for f in os.listdir(dirname):
            if f.endswith(".csv"):
                newfname = f[:-4] + ".custom"
                os.rename(os.path.join(dirname, f), os.path.join(dirname, newfname))
    bodo.barrier()
    try:
        check_func(test_impl, (fname2,), py_output=py_output)
    finally:
        if bodo.get_rank() == 0:
            for f in os.listdir(fname2):
                if f.endswith(".custom"):
                    newfname = f[:-7] + ".csv"
                    os.rename(os.path.join(dirname, f), os.path.join(dirname, newfname))
        bodo.barrier()


def test_csv_header_write_read(datapath, memory_leak_check):
    """Test writing and reading csv outputs containing headers"""

    df = pd.read_csv(datapath("example.csv"))
    pd_fname = "pd_csv_header_test.csv"

    def write_impl(df, fname):
        df.to_csv(fname)

    def read_impl(fname):
        return pd.read_csv(fname)

    if bodo.get_rank() == 0:
        write_impl(df, pd_fname)

    bodo.barrier()
    pd_res = read_impl(pd_fname)

    bodo_seq_write = bodo.jit(write_impl)
    bodo_1D_write = bodo.jit(all_args_distributed_block=True)(write_impl)
    bodo_1D_var_write = bodo.jit(all_args_distributed_varlength=True)(write_impl)
    arg_seq = (bodo_seq_write, df, "bodo_csv_header_test_seq.csv")
    arg_1D = (bodo_1D_write, _get_dist_arg(df, False), "bodo_csv_header_test_1D.csv")
    arg_1D_var = (
        bodo_1D_var_write,
        _get_dist_arg(df, False, True),
        "bodo_csv_header_test_1D_var.csv",
    )
    args = [arg_seq, arg_1D, arg_1D_var]
    for (func, df_arg, fname_arg) in args:
        with ensure_clean(fname_arg):
            func(df_arg, fname_arg)
            check_func(read_impl, (fname_arg,), py_output=pd_res)

    bodo.barrier()
    if bodo.get_rank() == 0:
        os.remove(pd_fname)


cat_csv_dtypes = {
    "C1": pd.Int64Dtype(),
    "C2": pd.CategoricalDtype(["A", "B", "C"]),
    "C3": str,
}


def test_csv_cat1(datapath, memory_leak_check):
    fname = datapath("csv_data_cat1.csv")

    def test_impl(fname):
        ct_dtype = pd.CategoricalDtype(["A", "B", "C"])
        dtypes = {"C1": np.dtype("int32"), "C2": ct_dtype, "C3": str}
        df = pd.read_csv(fname, names=["C1", "C2", "C3"], dtype=dtypes)
        return df

    def test_impl2(fname):
        df = pd.read_csv(fname, names=["C1", "C2", "C3"], dtype=cat_csv_dtypes)
        return df

    check_func(test_impl, (fname,))
    check_func(test_impl2, (fname,))

    compressed_names = compress_file(fname)
    try:
        for fname in compressed_names:
            check_func(test_impl, (fname,))
    finally:
        remove_files(compressed_names)


def test_csv_date_col_name(datapath, memory_leak_check):
    """Test the use of column names in "parse_dates" of read_csv"""
    fname = datapath("csv_data_date1.csv")

    def test_impl(fname):
        return pd.read_csv(
            fname,
            names=["A", "B", "C", "D"],
            dtype={"A": int, "B": float, "C": str, "D": int},
            parse_dates=["C"],
        )

    check_func(test_impl, (fname,))

    compressed_names = compress_file(fname)
    try:
        for fname in compressed_names:
            check_func(test_impl, (fname,))
    finally:
        remove_files(compressed_names)


def test_csv_read_only_datetime1(datapath, memory_leak_check):
    """Test the use of reading dataframe containing
    single datetime like column
    """
    fname = datapath("csv_data_only_date1.csv")

    def test_impl(fname):
        return pd.read_csv(
            fname,
            names=["A"],
            dtype={"A": str},
            parse_dates=["A"],
        )

    check_func(test_impl, (fname,))

    compressed_names = compress_file(fname)
    try:
        for fname in compressed_names:
            check_func(test_impl, (fname,))
    finally:
        remove_files(compressed_names)


def test_csv_read_only_datetime2(datapath, memory_leak_check):
    """Test the use of reading dataframe containing
    only datetime-like columns
    """
    fname = datapath("csv_data_only_date2.csv")

    def test_impl(fname):
        return pd.read_csv(
            fname,
            names=["A", "B"],
            dtype={"A": str, "B": str},
            parse_dates=[0, 1],
        )

    check_func(test_impl, (fname,))

    compressed_names = compress_file(fname)
    try:
        for fname in compressed_names:
            check_func(test_impl, (fname,))
    finally:
        remove_files(compressed_names)


def test_csv_dir_int_nulls_single(datapath, memory_leak_check):
    """
    Test read_csv reading dataframe containing int column with nulls
    from a directory(Spark output) containing single csv file.
    """
    fname = datapath("int_nulls_single.csv")

    def test_impl(fname):
        return pd.read_csv(fname, names=["A"], dtype={"A": "Int32"}, header=None)

    py_output = pd.read_csv(
        datapath("int_nulls.csv"), names=["A"], dtype={"A": "Int32"}, header=None
    )

    check_func(test_impl, (fname,), py_output=py_output)

    compress_dir(fname)
    try:
        check_func(test_impl, (fname,), py_output=py_output)
    finally:
        uncompress_dir(fname)


def test_csv_dir_int_nulls_header_single(datapath, memory_leak_check):
    """
    Test read_csv reading dataframe containing int column with nulls
    from a directory(Spark output) containing single csv file with header.
    """
    fname = datapath("int_nulls_header_single.csv")

    def test_impl(fname):
        return pd.read_csv(fname)

    # index_col = 0 because int_nulls.csv has index written
    # names=["A"] because int_nulls.csv does not have header
    py_output = pd.read_csv(datapath("int_nulls.csv"), index_col=0, names=["A"])

    check_func(test_impl, (fname,), py_output=py_output)

    compress_dir(fname)
    try:
        check_func(test_impl, (fname,), py_output=py_output)
    finally:
        uncompress_dir(fname)


def test_csv_dir_int_nulls_multi(datapath, memory_leak_check):
    """
    Test read_csv reading dataframe containing int column with nulls
    from a directory(Spark output) containing multiple csv file.
    """
    fname = datapath("int_nulls_multi.csv")

    def test_impl(fname):
        return pd.read_csv(
            fname,
            names=["A"],
            dtype={"A": "Int32"},
        )

    py_output = pd.read_csv(
        datapath("int_nulls.csv"),
        names=["A"],
        dtype={"A": "Int32"},
    )

    check_func(test_impl, (fname,), py_output=py_output)

    compress_dir(fname)
    try:
        check_func(test_impl, (fname,), py_output=py_output)
    finally:
        uncompress_dir(fname)


def test_csv_dir_int_nulls_header_multi(datapath, memory_leak_check):
    """
    Test read_csv reading dataframe containing int column with nulls
    from a directory(Spark output) containing multiple csv files with header,
    wtih infer dtypes.
    """
    fname = datapath("int_nulls_header_multi.csv")

    def test_impl(fname):
        return pd.read_csv(fname)

    # index_col = 0 because int_nulls.csv has index written
    # names=["A"] because int_nulls.csv does not have header
    py_output = pd.read_csv(datapath("int_nulls.csv"), index_col=0, names=["A"])

    check_func(test_impl, (fname,), py_output=py_output)

    compress_dir(fname)
    try:
        check_func(test_impl, (fname,), py_output=py_output)
    finally:
        uncompress_dir(fname)


def test_csv_dir_str_arr_single(datapath, memory_leak_check):
    """
    Test read_csv reading dataframe containing str column with nulls
    from a directory(Spark output) containing single csv file.
    """
    fname = datapath("str_arr_single.csv")

    def test_impl(fname):
        return pd.read_csv(
            fname,
            names=["A", "B"],
            dtype={"A": str, "B": str},
        ).fillna("")

    py_output = pd.read_csv(
        datapath("str_arr.csv"),
        names=["A", "B"],
        dtype={"A": str, "B": str},
    ).fillna("")

    check_func(test_impl, (fname,), py_output=py_output)

    compress_dir(fname)
    try:
        check_func(test_impl, (fname,), py_output=py_output)
    finally:
        uncompress_dir(fname)


def test_csv_dir_str_arr_multi(datapath, memory_leak_check):
    """
    Test read_csv reading dataframe containing str column with nulls
    from a directory(Spark output) containing multiple csv file.
    """
    fname = datapath("str_arr_parts.csv")

    def test_impl(fname):
        return pd.read_csv(
            fname,
            names=["A", "B"],
            dtype={"A": str, "B": str},
        ).fillna("")

    py_output = pd.read_csv(
        datapath("str_arr.csv"),
        names=["A", "B"],
        dtype={"A": str, "B": str},
    ).fillna("")

    check_func(test_impl, (fname,), py_output=py_output)

    compress_dir(fname)
    try:
        check_func(test_impl, (fname,), py_output=py_output)
    finally:
        uncompress_dir(fname)


@pytest.mark.smoke
def test_excel1(datapath, memory_leak_check):
    """Test pd.read_excel()"""

    def test_impl1(fname):
        return pd.read_excel(fname, parse_dates=[2])

    def test_impl2(fname):
        dtype = {
            "A": int,
            "B": float,
            "C": np.dtype("datetime64[ns]"),
            "D": str,
            "E": np.bool_,
        }
        return pd.read_excel(
            fname,
            sheet_name="Sheet1",
            parse_dates=["C"],
            dtype=dtype,
            names=list(dtype.keys()),
        )

    def test_impl3(fname):
        return pd.read_excel(fname, parse_dates=[2], comment="#")

    def test_impl4(fname, sheet):
        return pd.read_excel(fname, sheet, parse_dates=[2])

    def test_impl5(fname):
        dtype = {
            "A": int,
            "B": float,
            "C": np.dtype("datetime64[ns]"),
            "D": str,
            "E": np.bool_,
        }
        return pd.read_excel(
            fname,
            sheet_name="Sheet1",
            parse_dates=["C"],
            dtype=dtype,
        )

    # passing file name as argument to exercise value-based dispatch
    fname = datapath("data.xlsx")
    check_func(test_impl1, (fname,), is_out_distributed=False)
    check_func(test_impl2, (fname,), is_out_distributed=False)
    fname = datapath("data_comment.xlsx")
    assert pandas_version in (
        (1, 3),
        (1, 4),
    ), "`name` na-filtering issue for 1.4, check if it's fixed in later versions"
    if pandas_version == (1, 3):
        check_func(test_impl3, (fname,), is_out_distributed=False)
    fname = datapath("data.xlsx")
    check_func(test_impl4, (fname, "Sheet1"), is_out_distributed=False)
    with pytest.raises(BodoError, match="both 'dtype' and 'names' should be provided"):
        bodo.jit(test_impl5)(fname)


def test_csv_dtype_unicode(memory_leak_check):
    """
    Tests read_csv using dtype="unicode"
    """

    def test_impl(fname):
        return pd.read_csv(fname, names=["A", "B", "C", "D"], dtype="unicode")

    fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")
    check_func(test_impl, (fname,))


def _check_filenotfound(fname, func):
    with pytest.raises(BodoError) as excinfo:
        bodo.jit(func)(fname)
    err_track = excinfo.getrepr(style="native")
    assert "Pseudo-exception" not in str(err_track)


def test_file_not_found(memory_leak_check):
    """Test removal Pseduo-exception with FileNotFoundError"""

    def test_csv(fname):
        df = pd.read_csv(fname)
        return df.C

    def test_pq(fname):
        df = pd.read_parquet(fname)
        return len(df)

    def test_json(fname):
        df = pd.read_json(fname)
        return df.C

    _check_filenotfound("nofile.csv", test_csv)
    _check_filenotfound("nofile.pq", test_pq)
    _check_filenotfound("nofile.json", test_json)
    _check_filenotfound("s3://bodo-test/csv_data_date_not_found.csv", test_csv)
    _check_filenotfound(
        "gcs://anaconda-public-data/nyc-taxi/nyc.parquet/art.0.parquet", test_pq
    )


@pytest.mark.slow
def test_csv_relative_path(datapath, memory_leak_check):
    """Test pd.read_csv with relative path"""

    # File
    filename = os.path.join(".", "bodo", "tests", "data", "example.csv")

    def impl1():
        return pd.read_csv(filename)

    check_func(impl1, (), check_dtype=False)

    # Folder
    foldername = os.path.join(".", "bodo", "tests", "data", "example_multi.csv")
    py_output = pd.read_csv(datapath("example.csv"))

    def impl1():
        return pd.read_csv(foldername)

    check_func(impl1, (), check_dtype=False, py_output=py_output)


def test_read_csv_dict_encoded_string_arrays(datapath, memory_leak_check):
    """
    Test reading string arrays as dictionary-encoded in read_csv when specified by the
    user
    """
    fname = datapath("example.csv")

    # all string data as dict-encoded, dead column elimination
    def impl1(fname):
        df = pd.read_csv(fname, _bodo_read_as_dict=["two", "five"])
        return df[["two", "four", "five"]]

    py_output = pd.read_csv(fname)[["two", "four", "five"]]
    check_func(impl1, (fname,), py_output=py_output)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit()(impl1)(fname)
        check_logger_msg(stream, "Columns ['two', 'five'] using dictionary encoding")

    # only one string column as dict-encoded
    def impl2(fname):
        df = pd.read_csv(fname, _bodo_read_as_dict=["two"])
        return df[["one", "two", "five"]]

    py_output = pd.read_csv(fname)[["one", "two", "five"]]
    check_func(impl2, (fname,), py_output=py_output)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit()(impl2)(fname)
        check_logger_msg(stream, "Columns ['two'] using dictionary encoding")

    # error checking _bodo_read_as_dict
    with pytest.raises(BodoError, match=r"must be a constant list of column names"):

        def impl4(fname):
            df = pd.read_csv(fname, _bodo_read_as_dict=True)
            return df

        bodo.jit(impl4)(fname)

    with pytest.raises(BodoError, match=r"_bodo_read_as_dict is not in data columns"):

        def impl5(fname):
            df = pd.read_csv(fname, _bodo_read_as_dict=["H"])
            return df

        bodo.jit(impl5)(fname)

    with pytest.raises(BodoError, match=r"is not a string column"):

        def impl6(fname):
            df = pd.read_csv(fname, _bodo_read_as_dict=["one"])
            return df

        bodo.jit(impl6)(fname)


@pytest.mark.slow
def test_csv_nrows(memory_leak_check):
    """Test pd.read_csv with nrows argument"""
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl1():
        return pd.read_csv(fname, nrows=2)

    check_func(impl1, (), check_dtype=False)

    # Test nrows as variable.
    nrows = 6

    def impl2():
        return pd.read_csv(fname, nrows=nrows)

    check_func(impl2, (), check_dtype=False)

    # Test nrows + skiprows
    def impl3():
        return pd.read_csv(fname, skiprows=2, nrows=4)

    check_func(impl3, (), check_dtype=False)

    # Test with names provided
    fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

    def impl4():
        df = pd.read_csv(
            fname,
            names=["A", "B", "C", "D"],
            dtype={"A": int, "B": float, "C": float, "D": int},
            nrows=2,
            skiprows=1,
        )
        return df.B.values

    check_func(impl4, (), check_dtype=False)


@pytest.mark.slow
def test_csv_skiprows_var(memory_leak_check):
    """Test pd.read_csv with skiprows argument as variable"""
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    skip = 3

    def impl_skip():
        ans = pd.read_csv(fname, skiprows=skip)
        return ans

    check_func(impl_skip, (), check_dtype=False)

    # Test skiprows using loop index variable
    # set dtype=str since Bodo sets uniform type for the whole column
    # while Pandas sets type per chunk, resulting in miexed type output
    def impl_loop():
        nrows = 3
        colnames = ["A", "B", "C", "D", "E"]
        df_all = []
        for i in range(4):
            df = pd.read_csv(
                fname, skiprows=(i * nrows), nrows=nrows, names=colnames, dtype=str
            )
            df_all.append(df)
        return pd.concat(df_all)

    check_func(impl_loop, (), check_dtype=False, sort_output=True)


@pytest.mark.slow
def test_csv_chunksize_forloop(datapath, memory_leak_check):
    """
    Check the pd.read_csv iterator using chunksize and a for loop.
    """
    fname = datapath("example.csv")

    # This test checks array analysis generated shape nodes for this iterator
    # structure also, since .max() produces a parfor. Array analysis
    # modifies the IR output for the iterator. Here is an example:
    # $20for_iter.1 = iternext(value=$18get_iter.7) ['$18get_iter.7', '$20for_iter.1']
    # $val.156 = pair_first(value=$20for_iter.1) ['$20for_iter.1', '$val.156']
    # $20for_iter.2_shape.149 = getattr(value=$val.156, attr=shape) ['$20for_iter.2_shape.149', '$val.156']
    # $20for_iter.2_size0.150 = static_getitem(value=$20for_iter.2_shape.149, index=0, index_var=None, fn=<built-in function getitem>) ['$20for_iter.2_shape.149', '$20for_iter.2_size0.150']
    # $20for_iter.3 = pair_second(value=$20for_iter.1) ['$20for_iter.1', '$20for_iter.3']
    # branch $20for_iter.3, 22, 234            ['$20for_iter.3']
    #
    # This means that even once iterations are finished, the IR attempts to check
    # the shape value of the DataFrame which is a null value (potential segfault
    # if not handled properly).
    #
    # We handle this situation in our implementation of len(df), which is what gets produced
    # when shape is called. Here we check if the meminfo pointer is null and if so return
    # a garbage value (0) rather than looking at the data. We can do this because we assume
    # Numba 0 initializes structs with no provided value. Here are the relevant source code lines.
    #
    # https://github.com/numba/numba/blob/3131959c98e567d74ab6db402230cfea6ceecafe/numba/core/imputils.py#L335
    # https://github.com/numba/numba/blob/3131959c98e567d74ab6db402230cfea6ceecafe/numba/core/base.py#L987
    # https://github.com/numba/numba/blob/3131959c98e567d74ab6db402230cfea6ceecafe/numba/core/cgutils.py#L56
    # https://github.com/numba/numba/blob/3131959c98e567d74ab6db402230cfea6ceecafe/numba/core/cgutils.py#L130
    # Note the zfill=True in the last link should show that it is 0 initialized.

    def impl1(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=3):
            result = val["four"].max()
            total += result
        return total

    def impl2(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=100):
            # Single chunk
            result = val["four"].max()
            total += result
        return total

    def impl3(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=1):
            # Empty data on all ranks but 0
            result = val["four"].max()
            total += result
        return total

    check_func(impl1, (fname,))
    check_func(impl2, (fname,))
    check_func(impl3, (fname,))


@pytest.mark.slow
def test_csv_chunksize_index_col(datapath, memory_leak_check):
    """
    Check the pd.read_csv iterator using chunksize and an index_col.
    """
    fname = datapath("example.csv")

    def impl(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=3, index_col="four"):
            total += val.index[-1]
        return total

    check_func(impl, (fname,))


@pytest.mark.slow
def test_csv_chunksize_enumerate(datapath, memory_leak_check):
    """
    Check the pd.read_csv iterator using chunksize and enumerate.
    """
    fname = datapath("example.csv")

    def impl1(fname):
        total = 0.0
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=3)):
            result = val["four"].max()
            total += result
            count_total += i
        return (count_total, total)

    def impl2(fname):
        total = 0.0
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=100)):
            # Single chunk
            result = val["four"].max()
            total += result
            count_total += i
        return (count_total, total)

    def impl3(fname):
        total = 0.0
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=1)):
            # Empty data on all ranks but 0
            result = val["four"].max()
            total += result
            count_total += i
        return (count_total, total)

    check_func(impl1, (fname,))
    check_func(impl2, (fname,))
    check_func(impl3, (fname,))


@pytest.mark.slow
def test_csv_chunksize_forloop_append(datapath, memory_leak_check):
    """
    Check returning a dataframe with the pd.read_csv iterator using
    chunksize and a for loop.
    """

    fname = datapath("example.csv")

    def impl1(fname):
        df_list = []
        for val in pd.read_csv(fname, chunksize=3):
            df_list.append(val)
        return pd.concat(df_list)

    def impl2(fname):
        df_list = []
        for val in pd.read_csv(fname, chunksize=100):
            # Single chunk
            df_list.append(val)
        return pd.concat(df_list)

    def impl3(fname):
        df_list = []
        for val in pd.read_csv(fname, chunksize=1):
            # Empty data on all ranks but 0
            df_list.append(val)
        return pd.concat(df_list)

    # Check only sequential implementation because previous tests check
    # situations where the csv read is parallel. Returning a sequential
    # output should enable checking read_csv with parallel=False
    check_func(impl1, (fname,), is_out_distributed=False, dist_test=False)
    check_func(impl2, (fname,), is_out_distributed=False, dist_test=False)
    check_func(impl3, (fname,), is_out_distributed=False, dist_test=False)


@pytest.mark.slow
def test_csv_chunksize_enumerate_append(datapath, memory_leak_check):
    """
    Check returning a dataframe with the pd.read_csv iterator using
    chunksize and enumerate.
    """

    fname = datapath("example.csv")

    def impl1(fname):
        df_list = []
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=3)):
            df_list.append(val)
            count_total += i
        return (count_total, pd.concat(df_list))

    def impl2(fname):
        df_list = []
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=100)):
            # Single chunk
            df_list.append(val)
            count_total += i
        return (count_total, pd.concat(df_list))

    def impl3(fname):
        df_list = []
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=1)):
            # Empty data on all ranks but 0
            df_list.append(val)
            count_total += i
        return (count_total, pd.concat(df_list))

    # Check only sequential implementation because previous tests check
    # situations where the csv read is parallel. Returning a sequential
    # output should enable checking read_csv with parallel=False
    check_func(impl1, (fname,), is_out_distributed=False, dist_test=False)
    check_func(impl2, (fname,), is_out_distributed=False, dist_test=False)
    check_func(impl3, (fname,), is_out_distributed=False, dist_test=False)


@pytest.mark.slow
def test_csv_chunksize_forloop_nested(datapath, memory_leak_check):
    """
    Check multiple calls to the pd.read_csv iterator using
    chunksize and a for loop.
    """

    fname = datapath("example.csv")

    def impl1(fname, n):
        total = 0.0
        for _ in range(n):
            for val in pd.read_csv(fname, chunksize=3):
                result = val["four"].max()
                total += result
        return total

    def impl2(fname, n):
        total = 0.0
        for _ in range(n):
            for val in pd.read_csv(fname, chunksize=100):
                # Single chunk
                result = val["four"].max()
                total += result
        return total

    def impl3(fname, n):
        total = 0.0
        for _ in range(n):
            for val in pd.read_csv(fname, chunksize=1):
                # Empty data on all ranks but 0
                result = val["four"].max()
                total += result
        return total

    check_func(impl1, (fname, 5))
    check_func(impl2, (fname, 5))
    check_func(impl3, (fname, 5))


@pytest.mark.slow
def test_csv_chunksize_enumerate_nested(datapath, memory_leak_check):
    """
    Check multiple calls to the pd.read_csv iterator using
    chunksize and enumerate.
    """

    fname = datapath("example.csv")

    def impl1(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=3)):
                result = val["four"].max()
                total += result
                count_total += i
        return (count_total, total)

    def impl2(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=100)):
                # Single chunk
                result = val["four"].max()
                total += result
                count_total += i
        return (count_total, total)

    def impl3(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=1)):
                # Empty data on all ranks but 0
                result = val["four"].max()
                total += result
                count_total += i
        return (count_total, total)

    check_func(impl1, (fname, 5))
    check_func(impl2, (fname, 5))
    check_func(impl3, (fname, 5))


@pytest.mark.slow
def test_csv_chunksize_skiprows(datapath, memory_leak_check):
    """
    Check multiple calls to the pd.read_csv iterator using
    chunksize with skiprows
    """

    fname = datapath("example.csv")

    # skiprows and multiple chunks
    def impl1(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=3, skiprows=5)):
                result = val.iloc[:, 0].max()
                total += result
                count_total += i
        return (count_total, total)

    # skiprows and one chunk
    def impl2(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=100, skiprows=5)):
                # Single chunk
                result = val.iloc[:, 0].max()
                total += result
                count_total += i
        return (count_total, total)

    # skiprows all rows but one with one chunk on one rank only
    def impl3(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=1, skiprows=13)):
                # Empty data on all ranks but 0
                result = val.iloc[:, 0].max()
                total += result
                count_total += i
        return (count_total, total)

    check_func(impl1, (fname, 5))
    check_func(impl2, (fname, 5))
    check_func(impl3, (fname, 5))


@pytest.mark.slow
def test_csv_chunksize_nrows(datapath, memory_leak_check):
    """
    Check multiple calls to the pd.read_csv iterator using
    chunksize with nrows
    """

    fname = datapath("example.csv")

    # nrows and multiple chunks
    def impl1(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=3, nrows=7)):
                result = val.iloc[:, 0].max()
                total += result
                count_total += i
        return (count_total, total)

    # nrows and one chunk
    def impl2(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=100, nrows=5)):
                # Single chunk
                result = val.iloc[:, 0].max()
                total += result
                count_total += i
        return (count_total, total)

    # nrows and chunk on one rank only
    def impl3(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=1, nrows=3)):
                # Empty data on all ranks but 0
                result = val.iloc[:, 0].max()
                total += result
                count_total += i
        return (count_total, total)

    check_func(impl1, (fname, 5))
    check_func(impl2, (fname, 5))
    check_func(impl3, (fname, 5))


@pytest.mark.slow
def test_csv_chunksize_nrows_skiprows(datapath, memory_leak_check):
    """
    Check the pd.read_csv iterator using chunksize with nrows/skiprows
    """
    fname = datapath("example.csv")

    # nrows, skiprows and multiple chunks
    def impl1(fname):
        total = 0.0
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=3, nrows=10, skiprows=5)):
            result = val.iloc[:, 0].max()
            total += result
            count_total += i
        return (count_total, total)

    # nrows, skiprows and one chunk
    def impl2(fname):
        total = 0.0
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=100, skiprows=4, nrows=5)):
            # Single chunk
            result = val.iloc[:, 0].max()
            total += result
            count_total += i
        return (count_total, total)

    # nrows, skiprows and all data on one rank only
    def impl3(fname):
        total = 0.0
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=1, nrows=10, skiprows=5)):
            # Empty data on all ranks but 0
            result = val.iloc[:, 0].max()
            total += result
            count_total += i
        return (count_total, total)

    check_func(impl1, (fname,))
    check_func(impl2, (fname,))
    check_func(impl3, (fname,))


@pytest.mark.slow
def test_csv_skiprows_list_low_memory(datapath, memory_leak_check):
    """
    Test pd.read_csv skiprows as list for low_memory path
    """
    fname = datapath("example.csv")

    # skiprows list
    # (include 1st row and case where rank's start_pos is inside a skipped row)
    def impl1(fname):
        return pd.read_csv(fname, low_memory=True, skiprows=[1, 5, 7, 12])

    check_func(impl1, (fname,))

    # skiprows > available rows
    def impl2(fname):
        return pd.read_csv(fname, low_memory=True, skiprows=[17, 20])

    check_func(impl2, (fname,))

    # skiprows list unordered and duplicated
    def impl3(fname):
        return pd.read_csv(fname, low_memory=True, skiprows=[13, 4, 12, 4, 13])

    check_func(impl3, (fname,))

    # nrows + skiprows list (list has values in and out of nrows range)
    def impl4(fname):
        return pd.read_csv(fname, low_memory=True, nrows=10, skiprows=[4, 7, 12])

    check_func(impl4, (fname,))

    # skiprows list with 0
    def impl5(fname):
        return pd.read_csv(fname, low_memory=True, skiprows=[0, 2, 3, 4, 5, 6])

    check_func(impl5, (fname,))

    # contiguous list
    def impl6(fname):
        return pd.read_csv(fname, low_memory=True, skiprows=[6, 7, 8])

    check_func(impl6, (fname,))

    # header=None
    # Note: Bodo's index is object while pandas is int64
    # check_dtype=False does not work index values are '0' vs. 0
    py_output = pd.read_csv(
        fname, names=["0", "1", "2", "3", "4"], skiprows=[0, 2, 3, 7, 5, 9]
    )

    def impl7(fname):
        ans = pd.read_csv(fname, header=None, skiprows=[0, 2, 3, 7, 5, 9])
        return ans

    check_func(impl7, (fname,), py_output=py_output)


@pytest.mark.slow
def test_csv_skiprows_list(datapath, memory_leak_check):
    """
    Test pd.read_csv skiprows as list default case (i.e. low_memory=False)
    """
    fname = datapath("example.csv")

    # skiprows list
    # (include 1st row and case where rank's start_pos is inside a skipped row)
    def impl1(fname):
        return pd.read_csv(fname, low_memory=True, skiprows=[1, 5, 7, 12])

    check_func(impl1, (fname,))

    # skiprows > available rows
    def impl2(fname):
        return pd.read_csv(fname, skiprows=[17, 20])

    check_func(impl2, (fname,))

    # nrows + skiprows list (list has values in and out of nrows range)
    def impl3(fname):
        return pd.read_csv(fname, nrows=10, skiprows=[4, 7, 12])

    check_func(impl3, (fname,))

    # skiprows list with 0
    def impl4(fname):
        return pd.read_csv(fname, skiprows=[0, 2, 3, 4, 5, 6])

    check_func(impl4, (fname,))

    # header=None
    # Note: Bodo's index is object while pandas is int64
    # check_dtype=False does not work index values are '0' vs. 0
    py_output = pd.read_csv(
        fname, names=["0", "1", "2", "3", "4"], skiprows=[0, 2, 3, 7, 5, 9]
    )

    def impl5(fname):
        ans = pd.read_csv(fname, header=None, skiprows=[0, 2, 3, 7, 5, 9])
        return ans

    check_func(impl5, (fname,), py_output=py_output)


@pytest.mark.slow
def test_csv_skiprows_list_chunksize(datapath, memory_leak_check):
    """
    Test pd.read_csv skiprows as list with chunksize
    """
    fname = datapath("example.csv")
    # list + chunksize (each rank gets a chunk)
    # (include case where rank's chunk start_pos and end_pos is inside a skipped row)
    def impl1(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=5, skiprows=[4, 5, 7, 13]):
            result = val.iloc[:, 0].max()
            total += result
        return total

    check_func(impl1, (fname,))

    # list + chunksize (one chunk)
    def impl2(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=15, skiprows=[9, 4, 2, 10]):
            result = val.iloc[:, 0].max()
            total += result
        return total

    check_func(impl2, (fname,))

    # list + chunksize (only rank 0 gets the chunk)
    def impl3(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=1, skiprows=[4, 7]):
            result = val.iloc[:, 0].max()
            total += result
        return total

    check_func(impl3, (fname,))

    # list + nrows + chunksize (each rank gets a chunk)
    def impl4(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=3, nrows=12, skiprows=[2, 5, 8, 14]):
            result = val.iloc[:, 0].max()
            total += result
        return total

    check_func(impl4, (fname,))

    # list + nrows + chunksize (one chunk)
    def impl5(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=15, nrows=12, skiprows=[1, 4, 7, 13]):
            result = val.iloc[:, 0].max()
            total += result
        return total

    check_func(impl5, (fname,))

    # list + nrows + chunksize (only rank 0 gets the chunk)
    def impl6(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=1, nrows=6, skiprows=[14, 4, 9, 13]):
            result = val.iloc[:, 0].max()
            total += result
        return total

    check_func(impl6, (fname,))


@pytest.mark.slow
def test_csv_np_gt_rows(datapath, memory_leak_check):
    """Test when number of rows < number of ranks (np) with
    read_csv(). "small_data.csv" has one row.
    Running with np>1 will not fail
    """
    fname = datapath("small_data.csv")

    def impl1():
        return pd.read_csv(fname)

    check_func(impl1, (), check_dtype=False)

    # Test usecols
    def impl2():
        return pd.read_csv(fname, usecols=["A", "C"])

    check_func(impl2, (), check_dtype=False)


@pytest.mark.slow
def test_csv_escapechar(datapath, memory_leak_check):
    """Test pd.read_csv with escapechar argument"""

    fname = datapath("escapechar_data.csv")

    def impl():
        return pd.read_csv(fname, escapechar="\\")

    check_func(impl, (), check_dtype=False)


@pytest.mark.slow
def test_csv_unsupported_arg_match(memory_leak_check):
    """
    Test read_csv(): Test that passing an unsupported arg that matches
    the default doesn't raise an error
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        # squeeze is provided but not supported. It matches the default
        # so it should still work.
        return pd.read_csv(
            fname,
            ",",
            None,
            "infer",
            ["A", "B", "C", "D", "E"],
            None,
            None,
            False,
        )

    check_func(impl, (), check_dtype=False)


@pytest.mark.slow
def test_csv_unsupported_kwarg_match(memory_leak_check):
    """
    Test read_csv(): Test that passing an unsupported kwarg that matches
    the default doesn't raise an error
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        # comment is provided but not supported. It matches the default
        # so it should still work.
        return pd.read_csv(fname, comment=None)

    check_func(impl, (), check_dtype=False)


@pytest.mark.slow
def test_read_csv_sample_nrows(datapath, memory_leak_check):
    """Test read_csv with sample_nrows argument

    Args:
        datapath (fixture function): get path to test file
        memory_leak_check (fixture function): check memory leak in the test.

    """
    fname = datapath("large_data.csv")

    def impl1():
        return pd.read_csv(fname, sample_nrows=120)

    py_df = pd.read_csv(fname)
    check_func(impl1, (), py_output=py_df)


@pytest.mark.slow
def test_read_json_sample_nrows(datapath, memory_leak_check):
    """Test read_json with sample_nrows argument

    Args:
        datapath (fixture function): get path to test file
        memory_leak_check (fixture function): check memory leak in the test.

    """
    fname = datapath("large_data.json")

    def impl1():
        return pd.read_json(fname, sample_nrows=120)

    py_df = pd.read_json(fname, lines=True, orient="records")
    check_func(impl1, (), py_output=py_df)


@pytest.mark.slow
class TestIO(unittest.TestCase):
    def test_h5_write_parallel(self):
        fname = "lr_w.hdf5"

        def test_impl(N, D):
            points = np.ones((N, D))
            responses = np.arange(N) + 1.0
            f = h5py.File(fname, "w")
            dset1 = f.create_dataset("points", (N, D), dtype="f8")
            dset1[:] = points
            dset2 = f.create_dataset("responses", (N,), dtype="f8")
            dset2[:] = responses
            f.close()

        N = 101
        D = 10
        bodo_func = bodo.jit(test_impl)
        with ensure_clean(fname):
            bodo_func(N, D)
            f = h5py.File("lr_w.hdf5", "r")
            X = f["points"][:]
            Y = f["responses"][:]
            f.close()
            np.testing.assert_almost_equal(X, np.ones((N, D)))
            np.testing.assert_almost_equal(Y, np.arange(N) + 1.0)

    def test_h5_write_group(self):
        def test_impl(n, fname):
            arr = np.arange(n)
            n = len(arr)
            f = h5py.File(fname, "w")
            g1 = f.create_group("G")
            dset1 = g1.create_dataset("data", (n,), dtype="i8")
            dset1[:] = arr
            f.close()

        n = 101
        arr = np.arange(n)
        fname = "test_group.hdf5"
        bodo_func = bodo.jit(test_impl)
        with ensure_clean(fname):
            bodo_func(n, fname)
            f = h5py.File(fname, "r")
            X = f["G"]["data"][:]
            f.close()
            np.testing.assert_almost_equal(X, arr)

    def test_csv1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": float, "D": int},
            )

        check_func(test_impl, (), only_seq=True)

    def test_csv_keys1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            dtype = {"A": int, "B": float, "C": float, "D": int}
            return pd.read_csv(fname, names=list(dtype.keys()), dtype=dtype)

        check_func(test_impl, (), only_seq=True)

    def test_csv_const_dtype1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            dtype = {"A": "int", "B": "float64", "C": "float", "D": "int64"}
            return pd.read_csv(fname, names=list(dtype.keys()), dtype=dtype)

        check_func(test_impl, (), only_seq=True)

    def test_csv_infer1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_infer1.csv")

        def test_impl():
            return pd.read_csv(fname)

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_infer_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_infer1.csv")

        def test_impl():
            df = pd.read_csv(fname)
            return df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_csv_infer_str1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_cat1.csv")

        def test_impl():
            df = pd.read_csv(fname)
            return df

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_skip1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": float, "D": int},
                skiprows=2,
            )

        check_func(test_impl, (), only_seq=True)

    def test_csv_infer_skip1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_infer1.csv")

        def test_impl():
            return pd.read_csv(fname, skiprows=2)

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_infer_skip_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_infer1.csv")

        def test_impl():
            df = pd.read_csv(fname, skiprows=2, names=["A", "B", "C", "D"])
            return df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_csv_rm_dead1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            df = pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": float, "D": int},
            )
            return df.B.values

        check_func(test_impl, (), only_seq=True)

    def test_csv_date1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": str, "D": int},
                parse_dates=[2],
            )

        check_func(test_impl, (), only_seq=True)

    def test_csv_str1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": str, "D": int},
            )

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_index_name1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": str, "D": int},
                index_col="A",
            )

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_index_ind0(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": str, "D": int},
                index_col=0,
            )

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_index_ind1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": str, "D": int},
                index_col=1,
            )

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            df = pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": float, "D": int},
            )
            return (df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum())

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_csv_str_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            df = pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": str, "D": int},
            )
            return (df.A.sum(), df.B.sum(), (df.C == "1966-11-13").sum(), df.D.sum())

        bodo_func = bodo.jit(distributed=["df"])(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_csv_usecols1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            return pd.read_csv(fname, names=["C"], dtype={"C": float}, usecols=[2])

        check_func(test_impl, (), only_seq=True)

    def test_csv_usecols2(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            return pd.read_csv(fname, names=["B", "C"], usecols=[1, 2])

        check_func(test_impl, (), only_seq=True)

    def test_csv_usecols3(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data2.csv")

        def test_impl():
            return pd.read_csv(fname, sep="|", names=["B", "C"], usecols=[1, 2])

        check_func(test_impl, (), only_seq=True)

    def test_csv_cat2(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_cat1.csv")

        def test_impl():
            ct_dtype = pd.CategoricalDtype(["A", "B", "C", "D"])
            df = pd.read_csv(
                fname,
                names=["C1", "C2", "C3"],
                dtype={"C1": int, "C2": ct_dtype, "C3": str},
            )
            return df

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_single_dtype1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_dtype1.csv")

        def test_impl():
            df = pd.read_csv(fname, names=["C1", "C2"], dtype=np.float64)
            return df

        check_func(test_impl, (), only_seq=True)

    def write_csv(self, data_structure):
        # only run on a single processor
        if bodo.get_size() != 1:
            return

        def test_impl(data_structure, fname):
            data_structure.to_csv(fname)

        bodo_func = bodo.jit(test_impl)
        n = 111
        hp_fname = "test_write_csv1_bodo.csv"
        pd_fname = "test_write_csv1_pd.csv"
        with ensure_clean(pd_fname), ensure_clean(hp_fname):
            bodo_func(data_structure, hp_fname)
            test_impl(data_structure, pd_fname)
            pd.testing.assert_frame_equal(
                pd.read_csv(hp_fname), pd.read_csv(pd_fname), check_column_type=False
            )

    def test_write_dataframe_csv1(self):
        n = 111
        df = pd.DataFrame({"A": np.arange(n)}, index=np.arange(n) * 2)
        self.write_csv(df)

    def test_write_series_csv1(self):
        n = 111
        series = pd.Series(data=np.arange(n), dtype=np.float64, name="bodo")
        self.write_csv(series)

    def test_series_invalid_path_or_buf(self):
        n = 111
        series = pd.Series(data=np.arange(n), dtype=np.float64, name="bodo")

        def test_impl(data_structure, fname):
            data_structure.to_csv(fname)

        bodo_func = bodo.jit(test_impl)
        with pytest.raises(
            BodoError,
            match="argument should be None or string",
        ):
            bodo_func(series, 1)

    def write_csv_parallel(self, test_impl):
        bodo_func = bodo.jit(test_impl)
        n = 111
        hp_fname = "test_write_csv1_bodo_par.csv"
        pd_fname = "test_write_csv1_pd_par.csv"
        with ensure_clean(pd_fname), ensure_clean(hp_fname):
            bodo_func(n, hp_fname)
            self.assertEqual(count_array_REPs(), 0)
            self.assertEqual(count_parfor_REPs(), 0)
            if get_rank() == 0:
                test_impl(n, pd_fname)
                pd.testing.assert_frame_equal(
                    pd.read_csv(hp_fname),
                    pd.read_csv(pd_fname),
                    check_column_type=False,
                )

    def test_write_dataframe_csv_parallel1(self):
        def test_impl(n, fname):
            df = pd.DataFrame({"A": np.arange(n)})
            df.to_csv(fname)

        self.write_csv_parallel(test_impl)

    def test_write_series_csv_parallel1(self):
        def test_impl(n, fname):
            series = pd.Series(data=np.arange(n), dtype=np.float64, name="bodo")
            series.to_csv(fname)

        self.write_csv_parallel(test_impl)

    def test_write_csv_parallel2(self):
        # 1D_Var case
        def test_impl(n, fname):
            df = pd.DataFrame({"A": np.arange(n)})
            df = df[df.A % 2 == 1]
            df.to_csv(fname, index=False)

        bodo_func = bodo.jit(test_impl)
        n = 111
        hp_fname = "test_write_csv1_bodo_par.csv"
        pd_fname = "test_write_csv1_pd_par.csv"
        with ensure_clean(pd_fname), ensure_clean(hp_fname):
            bodo_func(n, hp_fname)
            self.assertEqual(count_array_REPs(), 0)
            self.assertEqual(count_parfor_REPs(), 0)
            if get_rank() == 0:
                test_impl(n, pd_fname)
                pd.testing.assert_frame_equal(
                    pd.read_csv(hp_fname),
                    pd.read_csv(pd_fname),
                    check_column_type=False,
                )


def check_CSV_write(
    write_impl,
    df,
    pandas_filename="pandas_out.csv",
    bodo_filename="bodo_out.csv",
    read_impl=None,
):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    # as of right now, only testing on posix, since it doesn't write to distributed file system
    # TODO: make this work for non posix
    if os.name != "posix":
        return
    if read_impl == None:
        read_impl = pd.read_csv

    n_pes = bodo.get_size()

    try:
        # only do the python CSV write on rank 0,
        # so the different ranks don't clobber each other.
        if bodo.get_rank() == 0:
            write_impl(df, pandas_filename)

        # copied from pq read/write section
        for mode in ["sequential", "1d-distributed", "1d-distributed-varlength"]:
            try:
                try:
                    if mode == "sequential":
                        bodo_write = bodo.jit(write_impl)
                        bodo_write(df, bodo_filename)
                    elif mode == "1d-distributed":
                        bodo_write = bodo.jit(
                            write_impl, all_args_distributed_block=True
                        )
                        bodo_write(_get_dist_arg(df, False), bodo_filename)
                    elif mode == "1d-distributed-varlength":
                        bodo_write = bodo.jit(
                            write_impl, all_args_distributed_varlength=True
                        )
                        bodo_write(_get_dist_arg(df, False, True), bodo_filename)
                    errors = comm.allgather(None)
                except Exception as e:
                    # In the case that one rank raises an exception, make sure that all the
                    # ranks raise an error, so we don't hang in the barrier.
                    comm.allgather(e)
                    raise

                for e in errors:
                    if isinstance(e, Exception):
                        raise e

                # wait until each rank has finished writing
                bodo.barrier()

                # read both files with pandas
                df1 = read_impl(pandas_filename)
                df2 = read_impl(bodo_filename)
                # read dataframes must be same as original except for dtypes
                passed = _test_equal_guard(
                    df1, df2, sort_output=False, check_names=True, check_dtype=False
                )
                n_passed = reduce_sum(passed)
                assert n_passed == n_pes
            finally:
                # cleanup the bodo file
                # TODO: update this if we use this test on non POSIX
                if bodo.get_rank() == 0:
                    try:
                        os.remove(bodo_filename)
                    except FileNotFoundError:
                        pass
    finally:
        # cleanup the pandas file
        if bodo.get_rank() == 0:
            try:
                os.remove(pandas_filename)
            except FileNotFoundError:
                pass


def check_to_csv_string_output(df, impl):
    """helper function that insures that output of to_csv is correct when returning a string from JIT code"""
    check_func(impl, (df,), only_seq=True)

    # check for distributed case, the output for each rank is the same as calling
    # df.to_csv(None) on the distributed dataframe
    py_out_1d = impl(_get_dist_arg(df))
    check_func(impl, (df,), only_1D=True, py_output=py_out_1d)

    py_out_1d_vars = impl(_get_dist_arg(df, var_length=True))
    check_func(impl, (df,), only_1DVar=True, py_output=py_out_1d_vars)


@pytest.mark.slow
def test_csv_non_constant_filepath_error(datapath):

    f1 = datapath("csv_data_cat1.csv")

    @bodo.jit
    def impl():
        for filepath in [f1]:
            df = pd.read_csv(
                filepath,
            )
        return df

    @bodo.jit
    def impl2():
        for filepath in [f1]:
            df = pd.read_csv(
                filepath,
                names=["A", "B", "C"],
            )
        return df

    @bodo.jit
    def impl3():
        for filepath in [f1]:
            df = pd.read_csv(
                filepath,
                dtype={
                    "A": int,
                    "B": str,
                    "C": str,
                },
            )
        return df

    def impl4():
        for filepath in [f1]:
            df = pd.read_csv(
                filepath,
                names=["A", "B", "C"],
                dtype={"A": int, "B": str, "C": str},
            )
        return df

    msg = (
        r".*pd.read_csv\(\) requires explicit type annotation using the 'names' "
        r"and 'dtype' arguments if the filename is not constant. For more information, "
        r"see: https://docs.bodo.ai/latest/file_io/#io_workflow."
    )

    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda: impl())()
    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda: impl2())()
    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda: impl3())()
    check_func(impl4, ())


@pytest.mark.slow
def test_json_non_constant_filepath_error(datapath):

    f1 = datapath("example.json")

    @bodo.jit
    def impl():
        for filepath in [f1]:
            df = pd.read_json(filepath, orient="records", lines=True)
        return df

    @bodo.jit
    def impl2():
        for filepath in [f1]:
            df = pd.read_json(
                filepath,
                orient="records",
                lines=True,
                dtype={
                    "one": float,
                    "two": str,
                    "three": np.bool_,
                    "four": float,
                    "five": str,
                },
            )
        return df

    @bodo.jit(
        locals={
            "df": {
                "one": bodo.float64[:],
                "two": bodo.string_array_type,
                "three": bodo.boolean_array,
                "four": bodo.float64[:],
                "five": bodo.string_array_type,
            }
        }
    )
    def impl3():
        for filepath in [f1]:
            df = pd.read_json(
                filepath,
                orient="records",
                lines=True,
            )
        return df

    msg = (
        r".*pd.read_json\(\) requires the filename to be a compile time constant. "
        r"For more information, see: https://docs.bodo.ai/latest/file_io/#json-section."
    )

    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda: impl())()
    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda: impl2())()
    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda: impl3())()


@pytest.mark.slow
def test_excel_non_constant_filepath_error(datapath):

    f1 = datapath("data.xlsx")

    @bodo.jit
    def impl():
        for filepath in [f1]:
            df = pd.read_excel(
                filepath,
            )
        return df

    @bodo.jit
    def impl2():
        for filepath in [f1]:
            df = pd.read_excel(
                filepath,
                names=["A", "B", "C", "D", "E"],
            )
        return df

    @bodo.jit
    def impl3():
        for filepath in [f1]:
            df = pd.read_excel(
                filepath,
                dtype={"A": int, "B": float, "C": str, "D": str, "E": np.bool_},
            )
        return df

    def impl4():
        for filepath in [f1]:
            df = pd.read_excel(
                filepath,
                names=["A", "B", "C", "D", "E"],
                dtype={"A": int, "B": float, "C": str, "D": str, "E": np.bool_},
            )
        return df

    msg1 = (
        r".*pd.read_excel\(\) requires explicit type annotation using the 'names' "
        r"and 'dtype' arguments if the filename is not constant. For more information, "
        r"see: https://docs.bodo.ai/latest/file_io/#io_workflow"
    )
    msg2 = r".*pd.read_excel\(\): both 'dtype' and 'names' should be provided if either is provided.*"

    with pytest.raises(BodoError, match=msg1):
        bodo.jit(lambda: impl())()
    with pytest.raises(BodoError, match=msg2):
        bodo.jit(lambda: impl2())()
    with pytest.raises(BodoError, match=msg2):
        bodo.jit(lambda: impl3())()
    # TODO [BE-1420]: Support distributed read_excel
    check_func(impl4, (), only_seq=True)


if __name__ == "__main__":
    unittest.main()
