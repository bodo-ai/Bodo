# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test I/O for JSON files using pd.read_json()
"""
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import _get_dist_arg, check_func
from bodo.utils.testing import ensure_clean, ensure_clean_dir
from bodo.utils.typing import BodoError


def compress_file(fname):
    assert not os.path.isdir(fname)
    if bodo.get_rank() == 0:
        subprocess.run(["gzip", "-k", "-f", fname])
        subprocess.run(["bzip2", "-k", "-f", fname])
    bodo.barrier()
    return [fname + ".gz", fname + ".bz2"]


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
            if f.endswith(".json") and os.path.getsize(dir_name + "/" + f) > 0
        ]:
            subprocess.run(["gzip", "-f", fname], cwd=dir_name)
    bodo.barrier()


def uncompress_dir(dir_name):
    if bodo.get_rank() == 0:
        for fname in [f for f in os.listdir(dir_name) if f.endswith(".gz")]:
            subprocess.run(["gunzip", fname], cwd=dir_name)
    bodo.barrier()


def test_read_json_list(datapath, memory_leak_check):
    """
    test read_json reads a dataframe containing columns with list of strings
    """
    fname_file = datapath("json_list_str.json")

    def test_impl(fname):
        return pd.read_json(fname, orient="records", lines=True)

    check_func(test_impl, (fname_file,))


def test_json_read_df(datapath, memory_leak_check):
    """
    test read_json reads a dataframe containing multiple columns
    from a single file, a directory containg a single json file,
    and a directory containg multiple json files
    """
    fname_file = datapath("example.json")
    fname_dir_single = datapath("example_single.json")
    fname_dir_multi = datapath("example_multi.json")

    def test_impl(fname):
        return pd.read_json(fname, orient="records", lines=True)

    def test_impl_with_dtype(fname):
        return pd.read_json(
            fname,
            orient="records",
            lines=True,
            dtype={
                "one": np.float32,
                "two": str,
                "three": "bool",
                "four": np.float32,
                "five": str,
            },
        )

    py_out = test_impl(fname_file)
    check_func(test_impl, (fname_file,), py_output=py_out)
    check_func(test_impl, (fname_dir_single,), py_output=py_out)
    # specify dtype here because small partition of dataframe causes only
    # int values(x.0) in float columns, and causes type mismatch because
    # pandas infers them as int columns
    check_func(test_impl_with_dtype, (fname_dir_multi,), py_output=py_out)

    compressed_names = compress_file(fname_file)
    try:
        for fname in compressed_names:
            check_func(test_impl, (fname,), py_output=py_out)
    finally:
        remove_files(compressed_names)

    compress_dir(fname_dir_single)
    try:
        check_func(test_impl, (fname_dir_single,), py_output=py_out)
    finally:
        uncompress_dir(fname_dir_single)

    compress_dir(fname_dir_multi)
    try:
        check_func(test_impl_with_dtype, (fname_dir_multi,), py_output=py_out)
    finally:
        uncompress_dir(fname_dir_multi)


@pytest.mark.slow
def test_json_read_int_nulls(datapath, memory_leak_check):
    """
    test read_json reads a dataframe containing nullable int column
    from a single file, a directory containg a single json file,
    and a directory containg multiple json files
    """
    fname_file = datapath("int_nulls.json")
    fname_dir_single = datapath("int_nulls_single.json")
    fname_dir_multi = datapath("int_nulls_multi.json")

    def test_impl(fname):
        return pd.read_json(fname, orient="records", lines=True)

    py_out = test_impl(fname_file)
    check_func(test_impl, (fname_file,), py_output=py_out)
    check_func(test_impl, (fname_dir_single,), py_output=py_out)
    check_func(test_impl, (fname_dir_multi,), py_output=py_out)


@pytest.mark.slow
def test_json_read_str_arr(datapath, memory_leak_check):
    """
    test read_json reads a dataframe containing str column
    from a single file, a directory containg a single json file,
    and a directory containg multiple json files
    """
    fname_file = datapath("str_arr.json")
    # Because spark and pandas writes null entries in json file differently
    # which causes different null values(None vs. nan) when pandas read them,
    # we pass this spark output file for pandas to read and use it as py_output
    fname_dir_file = datapath(
        "str_arr_single.json/part-00000-a0ff525c-31ec-499a-bde3-fe2a95cfbf8e-c000.json"
    )
    fname_dir_single = datapath("str_arr_single.json")
    fname_dir_multi = datapath("str_arr_parts.json")

    def test_impl(fname):
        return pd.read_json(
            # dtype required here, because pandas read string array as object type
            fname,
            orient="records",
            lines=True,
            dtype={"A": str, "B": str},
        )

    check_func(test_impl, (fname_file,))
    py_out = test_impl(fname_dir_file)
    check_func(test_impl, (fname_dir_single,), py_output=py_out)
    check_func(test_impl, (fname_dir_multi,), py_output=py_out)


@pytest.mark.smoke
def test_json_read_multiline_object(datapath, memory_leak_check):
    """
    test read_json where json object is multi-lined
    from a single file
    TODO: read a directory
    """
    fname = datapath("multiline_obj.json")

    def test_impl():
        return pd.read_json(
            fname,
            orient="records",
            lines=False,
            dtype={
                "RecordNumber": np.int64,
                "Zipcode": np.int64,
                "ZipCodeType": str,
                "City": str,
                "State": str,
            },
        )

    check_func(
        test_impl,
        (),
    )


def test_json_invalid_path_const(memory_leak_check):
    """test error raise when file path provided as constant but is invalid."""

    def test_impl():
        return pd.read_json("in_data_invalid.json")

    with pytest.raises(BodoError, match="No such file or directory"):
        bodo.jit(test_impl)()


def json_write_test(test_impl, read_impl, df, sort_col, reset_index=False):
    """
    A helper function used to test json write correctness
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    # get pandas output
    if bodo.get_rank() == 0:
        test_impl(df, "df_pd.json")
    bodo.barrier()
    pd_res = read_impl("df_pd.json")
    # bodo functions
    bodo_seq = bodo.jit(test_impl)
    bodo_1D = bodo.jit(all_args_distributed_block=True)(test_impl)
    bodo_1D_var = bodo.jit(all_args_distributed_varlength=True)(test_impl)
    # arguments for sequential, 1D distributed, and 1D var testing
    arg_seq = (bodo_seq, df, "df_seq.json")
    arg_1D = (bodo_1D, _get_dist_arg(df, False), "df_1d.json")
    arg_1D_var = (bodo_1D_var, _get_dist_arg(df, False, True), "df_1d_var.json")
    args = [arg_seq, arg_1D, arg_1D_var]
    # test writing sequentially, 1D distributed, and 1D Var-length
    for (func, df_arg, fname_arg) in args:
        with ensure_clean(fname_arg), ensure_clean_dir(fname_arg):
            func(df_arg, fname_arg)
            bodo.barrier()
            try:
                if bodo.get_rank() == 0:
                    if os.path.isfile(fname_arg):
                        pd.testing.assert_frame_equal(
                            read_impl(fname_arg), pd_res, check_column_type=False
                        )
                    else:
                        # pandas read single each json file in directory then concat
                        json_files = os.listdir(fname_arg)
                        assert len(json_files) > 0
                        results = [
                            read_impl(os.path.join(fname_arg, fname))
                            for fname in json_files
                        ]
                        bodo_res = pd.concat(
                            [
                                read_impl(os.path.join(fname_arg, fname))
                                for fname in json_files
                            ]
                        )
                        if reset_index:
                            pd.testing.assert_frame_equal(
                                bodo_res.sort_values(sort_col).reset_index(drop=True),
                                pd_res.sort_values(sort_col).reset_index(drop=True),
                                check_column_type=False,
                            )
                        else:
                            pd.testing.assert_frame_equal(
                                bodo_res.sort_values(sort_col),
                                pd_res.sort_values(sort_col),
                                check_column_type=False,
                            )
                errors = comm.allgather(None)
            except Exception as e:
                # The typing issue typically occurs on only a subset of processes,
                # because the process got a chunk of data from Python that is empty
                # or that we cannot currently type correctly.
                # To avoid a hang, we need to notify every rank of the error.
                comm.allgather(e)
                raise
            for e in errors:
                if isinstance(e, Exception):
                    raise e

    if bodo.get_rank() == 0:
        if os.path.exists("df_pd.json") and os.path.isfile("df_pd.json"):
            os.remove("df_pd.json")
    bodo.barrier()


@pytest.fixture(
    params=[
        pd.DataFrame(
            {
                "A": pd.date_range(start="2018-04-24", periods=12),
                "B": ["¡Y tú quién te crees?", "🐍⚡", "大处着眼，小处着手。", "hi", "a123", ""]
                * 2,
                "C": np.arange(12).astype(np.float64),
                "D": [True, False, np.nan, False, False, True] * 2,
            }
        )
    ]
)
def test_df(request, memory_leak_check):
    return request.param


@pytest.mark.smoke
def test_json_write_simple_df(memory_leak_check):
    """
    test to_json with default arguments
    Bodo has different default than Pandas for orient and lines
    This tests Pandas default
    """

    def test_impl(df, fname):
        df.to_json(fname, orient="columns", lines=False)

    def read_impl(fname):
        return pd.read_json(fname)

    n = 10
    df = pd.DataFrame(
        {
            "A": np.arange(n),
            "B": np.arange(n) % 2,
        },
        index=np.arange(n) * 2,
    )
    json_write_test(test_impl, read_impl, df, "A")


def test_json_write_simple_df_records(test_df, memory_leak_check):
    """
    test to_json with orient='records', lines=False
    """

    def test_impl(df, fname):
        df.to_json(fname, orient="records", lines=False)

    def read_impl(fname):
        # Supply D has a boolean dtype because there are null values.
        # Pandas by default will convert boolean with null values to float.
        return pd.read_json(
            fname, orient="records", lines=False, dtype={"D": "boolean"}
        )

    json_write_test(test_impl, read_impl, test_df, "C", reset_index=True)


def test_json_write_simple_df_records_lines(memory_leak_check):
    """
    test to_json with orient='records', lines=True
    """

    def test_impl(df, fname):
        df.to_json(fname, orient="records", lines=True)

    def read_impl(fname):
        return pd.read_json(fname, orient="records", lines=True)

    n = 100
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) % 15})
    json_write_test(test_impl, read_impl, df, "A")


@pytest.mark.parametrize("orient", ["split", "index", "columns", "table"])
def test_json_write_orient(test_df, orient, memory_leak_check):
    """
    test to_json with different orient options
    missing orient = "values" because only value arrays are written and
    thus difficult to test when we write to a directory
    """

    def test_impl(df, fname):
        df.to_json(fname, orient=orient, lines=False)

    def read_impl(fname):
        # Supply D has a boolean dtype because there are null values.
        # Pandas by default will convert boolean with null values to float.
        # This is true for all orients except table.
        if orient != "table":
            dtype = {"D": "boolean"}
        else:
            dtype = None

        return pd.read_json(fname, orient=orient, dtype=dtype)

    json_write_test(test_impl, read_impl, test_df, "C")


@pytest.mark.slow
def test_json_write_read_simple_df(memory_leak_check):
    """
    test to_json with default arguments
    matching read_json with default arguments for Bodo
    """

    def write_impl(n, fname):
        df = pd.DataFrame(
            {
                "A": np.arange(n),
                "B": np.arange(n) % 2,
            },
            index=np.arange(n) * 2,
        )
        df.to_json(fname)

    def read_impl(fname):
        return pd.read_json(fname)

    n = 10
    fname_file = "json_data.json"
    bodo.jit(write_impl)(n, fname_file)
    py_output = df = pd.DataFrame(
        {
            "A": np.arange(n),
            "B": np.arange(n) % 2,
        },
        index=np.arange(n) * 2,
    )
    check_func(
        read_impl,
        (fname_file,),
        py_output=py_output,
        reset_index=True,
        check_dtype=False,
    )
    if bodo.get_rank() == 0:
        if bodo.get_rank() == 0:
            shutil.rmtree(fname_file, ignore_errors=True)
