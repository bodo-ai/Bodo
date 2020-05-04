# Copyright (C) 2019 Bodo Inc.
import pytest
import pandas as pd
import numpy as np
import bodo
from bodo.tests.utils import check_func, _get_dist_arg, _test_equal_guard, reduce_sum

pytestmark = pytest.mark.hdfs


def test_hdfs_pq_groupby3(datapath, hdfs_datapath):
    """
    test hdfs read_parquet
    """

    hdfs_fname = hdfs_datapath("groupby3.pq")

    def test_impl(hdfs_fname):
        return pd.read_parquet(hdfs_fname)

    fname = datapath("groupby3.pq")
    py_output = pd.read_parquet(fname)

    check_func(test_impl, (hdfs_fname,), py_output=py_output)


def test_hdfs_pq_asof1(datapath, hdfs_datapath):
    """
    test hdfs read_parquet
    """

    hdfs_fname = hdfs_datapath("asof1.pq")

    def test_impl(hdfs_fname):
        return pd.read_parquet(hdfs_fname)

    fname = datapath("asof1.pq")
    py_output = pd.read_parquet(fname)

    check_func(test_impl, (hdfs_fname,), py_output=py_output)


def test_hdfs_pq_int_nulls_multi(datapath, hdfs_datapath):
    """
    test hdfs read_parquet of a directory containing multiple files
    """

    hdfs_fname = hdfs_datapath("int_nulls_multi.pq")

    def test_impl(hdfs_fname):
        return pd.read_parquet(hdfs_fname)

    fname = datapath("int_nulls_multi.pq")
    py_output = pd.read_parquet(fname)

    check_func(test_impl, (hdfs_fname,), py_output=py_output, check_dtype=False)


def test_csv_data1(datapath, hdfs_datapath):
    """
    test hdfs read_csv
    """

    hdfs_fname = hdfs_datapath("csv_data1.csv")

    def test_impl(hdfs_fname):
        return pd.read_csv(
            hdfs_fname,
            names=["A", "B", "C", "D"],
            dtype={"A": np.int, "B": np.float, "C": np.float, "D": np.int},
        )

    fname = datapath("csv_data1.csv")
    py_output = pd.read_csv(
        fname,
        names=["A", "B", "C", "D"],
        dtype={"A": np.int, "B": np.float, "C": np.float, "D": np.int},
    )

    check_func(test_impl, (hdfs_fname,), py_output=py_output)


def test_csv_data_date1(datapath, hdfs_datapath):
    """
    test hdfs read_csv
    """

    hdfs_fname = hdfs_datapath("csv_data_date1.csv")

    def test_impl(hdfs_fname):
        return pd.read_csv(
            hdfs_fname,
            names=["A", "B", "C", "D"],
            dtype={"A": np.int, "B": np.float, "C": str, "D": np.int},
            parse_dates=[2],
        )

    fname = datapath("csv_data_date1.csv")
    py_output = pd.read_csv(
        fname,
        names=["A", "B", "C", "D"],
        dtype={"A": np.int, "B": np.float, "C": str, "D": np.int},
        parse_dates=[2],
    )

    check_func(test_impl, (hdfs_fname,), py_output=py_output)


def test_hdfs_read_json(datapath, hdfs_datapath):
    """
    test read_json from hdfs
    """
    fname_file = hdfs_datapath("example.json")
    fname_dir_single = hdfs_datapath("example_single.json")
    fname_dir_multi = hdfs_datapath("example_multi.json")

    def test_impl(fname):
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

    py_out = test_impl(datapath("example.json"))
    check_func(test_impl, (fname_file,), py_output=py_out)
    check_func(test_impl, (fname_dir_single,), py_output=py_out)
    check_func(test_impl, (fname_dir_multi,), py_output=py_out)


@pytest.fixture(
    params=[
        pd.DataFrame(
            {
                "A": [1.1, np.nan, 4.2, 3.1, -1.3],
                "B": [True, False, False, True, True],
                "C": [1, 4, -5, -11, 6],
            }
        )
    ]
)
def test_df(request):
    return request.param


def test_hdfs_parquet_write_seq(hdfs_datapath, test_df):
    """
    test hdfs to_parquet sequentially
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_seq.pq")

    def test_write(test_df, hdfs_fname):
        test_df.to_parquet(hdfs_fname)

    bodo_write = bodo.jit(test_write)
    bodo_write(test_df, hdfs_fname)


def test_hdfs_parquet_write_1D(hdfs_datapath, test_df):
    """
    test hdfs to_parquet in 1D distributed
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_1D.pq")

    def test_write(test_df, hdfs_fname):
        test_df.to_parquet(hdfs_fname)

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False), hdfs_fname)


def test_hdfs_parquet_write_1D_var(hdfs_datapath, test_df):
    """
    test hdfs to_parquet in 1D Var distributed
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_1D_var.pq")

    def test_write(test_df, hdfs_fname):
        test_df.to_parquet(hdfs_fname)

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False, True), hdfs_fname)


def test_hdfs_csv_write_seq(hdfs_datapath, test_df):
    """
    test hdfs to_csv sequentially
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_seq.csv")

    def test_write(test_df, hdfs_fname):
        test_df.to_csv(hdfs_fname, index=False, header=False)

    bodo_write = bodo.jit(test_write)
    bodo_write(test_df, hdfs_fname)


def test_hdfs_csv_write_1D(hdfs_datapath, test_df):
    """
    test hdfs to_csv in 1D distributed
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_1D.csv")

    def test_write(test_df, hdfs_fname):
        test_df.to_csv(hdfs_fname, index=False, header=False)

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False), hdfs_fname)


def test_hdfs_csv_write_1D_var(hdfs_datapath, test_df):
    """
    test hdfs to_csv in 1D Var distributed
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_1D_var.csv")

    def test_write(test_df, hdfs_fname):
        test_df.to_csv(hdfs_fname, index=False, header=False)

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False, True), hdfs_fname)


def test_hdfs_json_write_records_lines_seq(hdfs_datapath, test_df):
    """
    test hdfs to_json(orient="records", lines=True) sequentially
    """

    hdfs_fname = hdfs_datapath("df_records_lines_seq.json")

    def test_write(test_df, hdfs_fname):
        test_df.to_json(hdfs_fname, orient="records", lines=True)

    bodo_write = bodo.jit(test_write)
    bodo_write(test_df, hdfs_fname)


def test_hdfs_json_write_records_lines_1D(hdfs_datapath, test_df):
    """
    test hdfs to_json(orient="records", lines=True) in 1D distributed
    """

    hdfs_fname = hdfs_datapath("df_records_lines_1D.json")

    def test_write(test_df, hdfs_fname):
        test_df.to_json(hdfs_fname, orient="records", lines=True)

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False), hdfs_fname)


def test_hdfs_json_write_records_lines_1D_var(hdfs_datapath, test_df):
    """
    test hdfs to_json(orient="records", lines=True) in 1D var
    """

    hdfs_fname = hdfs_datapath("df_records_lines_1D_var.json")

    def test_write(test_df, hdfs_fname):
        test_df.to_json(hdfs_fname, orient="records", lines=True)

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False, True), hdfs_fname)


def test_hdfs_parquet_read_seq(hdfs_datapath, test_df):
    """
    read_parquet 
    test the parquet file we just wrote sequentially
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_seq.pq")

    def test_read(hdfs_fname):
        return pd.read_parquet(hdfs_fname)

    check_func(test_read, (hdfs_fname,), py_output=test_df)


def test_hdfs_parquet_read_1D(hdfs_datapath, test_df):
    """
    read_parquet 
    test the parquet file we just wrote in 1D
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_1D.pq")

    def test_read(hdfs_fname):
        return pd.read_parquet(hdfs_fname)

    check_func(test_read, (hdfs_fname,), py_output=test_df)


def test_hdfs_parquet_read_1D_var(hdfs_datapath, test_df):
    """
    read_parquet 
    test the parquet file we just wrote in 1D Var
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_1D_var.pq")

    def test_read(hdfs_fname):
        return pd.read_parquet(hdfs_fname)

    check_func(test_read, (hdfs_fname,), py_output=test_df)


def test_hdfs_csv_read_seq(hdfs_datapath, test_df):
    """
    read_csv 
    test the csv file we just wrote sequentially
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_seq.csv")

    def test_read(hdfs_fname):
        return pd.read_csv(
            hdfs_fname,
            names=["A", "B", "C"],
            dtype={"A": np.float, "B": "bool", "C": np.int},
        )

    check_func(test_read, (hdfs_fname,), py_output=test_df)


def test_hdfs_csv_read_1D(hdfs_datapath, test_df):
    """
    read_csv 
    test the csv file we just wrote in 1D
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_1D.csv")

    def test_read(hdfs_fname):
        return pd.read_csv(
            hdfs_fname,
            names=["A", "B", "C"],
            dtype={"A": np.float, "B": "bool", "C": np.int},
        )

    check_func(test_read, (hdfs_fname,), py_output=test_df)


def test_hdfs_csv_read_1D_var(datapath, hdfs_datapath, test_df):
    """
    read_csv 
    test the csv file we just wrote in 1D Var
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_1D_var.csv")

    def test_read(hdfs_fname):
        return pd.read_csv(
            hdfs_fname,
            names=["A", "B", "C"],
            dtype={"A": np.float, "B": "bool", "C": np.int},
        )

    check_func(test_read, (hdfs_fname,), py_output=test_df)


@pytest.fixture(params=[np.arange(5)])
def test_np_arr(request):
    return request.param


def test_hdfs_np_tofile_seq(hdfs_datapath, test_np_arr):
    """
    test hdfs to_file
    """

    def test_write(test_np_arr, hdfs_fname):
        test_np_arr.tofile(hdfs_fname)

    hdfs_fname = hdfs_datapath("test_np_arr_bodo_seq.dat")
    bodo_func = bodo.jit(test_write)
    bodo_func(test_np_arr, hdfs_fname)


def test_hdfs_np_tofile_1D(hdfs_datapath, test_np_arr):
    """
    test hdfs to_file in 1D
    """

    def test_write(test_np_arr, hdfs_fname):
        test_np_arr.tofile(hdfs_fname)

    hdfs_fname = hdfs_datapath("test_np_arr_bodo_1D.dat")
    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_np_arr, False), hdfs_fname)


def test_hdfs_np_tofile_1D_var(hdfs_datapath, test_np_arr):
    """
    test hdfs to_file in 1D distributed
    """

    def test_write(test_np_arr, hdfs_fname):
        test_np_arr.tofile(hdfs_fname)

    hdfs_fname = hdfs_datapath("test_np_arr_bodo_1D_var.dat")
    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_np_arr, False, True), hdfs_fname)


def test_hdfs_np_fromfile_seq(hdfs_datapath, test_np_arr):
    """
    fromfile
    test the dat file we just wrote sequentially
    """

    def test_read(hdfs_fname):
        return np.fromfile(hdfs_fname, np.int64)

    hdfs_fname = hdfs_datapath("test_np_arr_bodo_seq.dat")
    check_func(test_read, (hdfs_fname,), py_output=test_np_arr)


def test_hdfs_np_fromfile_1D(hdfs_datapath, test_np_arr):
    """
    fromfile
    test the dat file we just wrote in 1D
    """

    def test_read(hdfs_fname):
        return np.fromfile(hdfs_fname, np.int64)

    hdfs_fname = hdfs_datapath("test_np_arr_bodo_1D.dat")
    check_func(test_read, (hdfs_fname,), py_output=test_np_arr, is_out_distributed=True)


def test_hdfs_np_fromfile_1D_var(hdfs_datapath, test_np_arr):
    """
    fromfile
    test the dat file we just wrote in 1D var
    """

    def test_read(hdfs_fname):
        return np.fromfile(hdfs_fname, np.int64)

    hdfs_fname = hdfs_datapath("test_np_arr_bodo_1D_var.dat")
    check_func(test_read, (hdfs_fname,), py_output=test_np_arr, is_out_distributed=True)


def test_hdfs_json_read_records_lines_seq(hdfs_datapath, test_df):
    """
    read_json(orient="records", lines=True)
    test the json file we just wrote sequentially
    """

    def test_read(hdfs_fname):
        return pd.read_json(
            hdfs_fname,
            orient="records",
            lines=True,
            dtype={"A": np.float, "B": "bool", "C": np.int},
        )

    hdfs_fname = hdfs_datapath("df_records_lines_seq.json")
    check_func(test_read, (hdfs_fname,), py_output=test_df)


def test_hdfs_json_read_records_lines_1D(hdfs_datapath, test_df):
    """
    read_json(orient="records", lines=True) 
    test the json file we just wrote in 1D
    """

    def test_read(hdfs_fname):
        return pd.read_json(
            hdfs_fname,
            orient="records",
            lines=True,
            dtype={"A": np.float, "B": "bool", "C": np.int},
        )

    hdfs_fname = hdfs_datapath("df_records_lines_1D.json")
    check_func(test_read, (hdfs_fname,), py_output=test_df)


def test_hdfs_json_read_records_lines_1D_var(hdfs_datapath, test_df):
    """
    read_json(orient="records", lines=True) 
    test the json file we just wrote in 1D Var
    """

    def test_read(hdfs_fname):
        return pd.read_json(
            hdfs_fname,
            orient="records",
            lines=True,
            dtype={"A": np.float, "B": "bool", "C": np.int},
        )

    hdfs_fname = hdfs_datapath("df_records_lines_1D_var.json")
    check_func(test_read, (hdfs_fname,), py_output=test_df)
