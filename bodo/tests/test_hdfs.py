# Copyright (C) 2019 Bodo Inc.

import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

import bodo
from bodo.tests.utils import _get_dist_arg, check_func
from bodo.utils.testing import ensure_clean2

pytestmark = pytest.mark.hdfs


def test_partition_cols(hdfs_datapath):
    """Test hdfs to_parquet partition_cols."""
    bd_fname = hdfs_datapath("bd_file.pq")
    pd_fname = hdfs_datapath("pd_file.pq")
    df = pd.DataFrame({"A": [0, 0, 0, 1, 1, 1], "B": [0, 1, 2, 3, 4, 5]})
    part_cols = ["A"]
    f = lambda df, part_cols: df.to_parquet(bd_fname, partition_cols=part_cols)
    with ensure_clean2(bd_fname), ensure_clean2(pd_fname):
        bodo.jit(f, distributed=["df"])(_get_dist_arg(df, False), part_cols)
        if bodo.get_rank() == 0:
            df.to_parquet(pd_fname, partition_cols=part_cols)
        bodo.barrier()
        bd_out = pd.read_parquet(bd_fname)
        pd_out = pd.read_parquet(pd_fname)
    pd.testing.assert_frame_equal(bd_out, pd_out, check_column_type=False)


def test_hdfs_write_parquet_no_empty_files(hdfs_datapath, memory_leak_check):
    """Test that when a rank has no data, it doesn't write a file"""
    # The test is most useful when run with multiple ranks
    # but should pass on a single rank too.
    from urllib.parse import urlparse

    from pyarrow.fs import FileSelector, FileType
    from pyarrow.fs import HadoopFileSystem as HdFS

    output_filename = hdfs_datapath("1row.pq")
    options = urlparse(output_filename)

    @bodo.jit(distributed=["df"])
    def impl(df, out_name):
        df.to_parquet(out_name)

    if bodo.get_rank() == 0:
        df = pd.DataFrame({"A": [1], "B": [1]})
    else:
        df = pd.DataFrame({"A": [], "B": []})

    with ensure_clean2(output_filename):
        impl(df, output_filename)
        bodo.barrier()
        # Only rank 0 should've written a file
        fs = HdFS(host=options.hostname, port=options.port, user=options.username)
        fi = fs.get_file_info(options.path)
        is_dir = (not fi.size) and fi.type == FileType.Directory
        assert is_dir, "Not a directory"
        file_selector = FileSelector(options.path, recursive=True)
        file_stats = fs.get_file_info(file_selector)
        assert len(file_stats) == 1


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


def test_hdfs_pq_trailing_sep(datapath, hdfs_datapath):
    """Test HDFS read_parquet on a directory works with a trailing sep"""
    hdfs_fname = hdfs_datapath("int_nulls_multi.pq/")

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
            dtype={"A": int, "B": float, "C": float, "D": int},
        )

    fname = datapath("csv_data1.csv")
    py_output = pd.read_csv(
        fname,
        names=["A", "B", "C", "D"],
        dtype={"A": int, "B": float, "C": float, "D": int},
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
            dtype={"A": int, "B": float, "C": str, "D": int},
            parse_dates=[2],
        )

    fname = datapath("csv_data_date1.csv")
    py_output = pd.read_csv(
        fname,
        names=["A", "B", "C", "D"],
        dtype={"A": int, "B": float, "C": str, "D": int},
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

    py_out = test_impl(datapath("example.json"))
    check_func(test_impl, (fname_file,), py_output=py_out)
    check_func(test_impl, (fname_dir_single,), py_output=py_out)
    # specify dtype here because small partition of dataframe causes only
    # int values(x.0) in float columns, and causes type mismatch becasue
    # pandas infer them as int columns
    check_func(test_impl_with_dtype, (fname_dir_multi,), py_output=py_out)


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

    comm = MPI.COMM_WORLD

    hdfs_fname = hdfs_datapath("test_df_bodo_seq.csv")

    def test_write(test_df, hdfs_fname):
        test_df.to_csv(hdfs_fname, index=False, header=False)

    bodo_write = bodo.jit(test_write)
    err = None
    if bodo.get_rank() == 0:
        try:
            bodo_write(test_df, hdfs_fname)
        except Exception as e:
            err = e
    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err


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


def test_hdfs_csv_write_header_seq(hdfs_datapath, test_df):
    """
    test hdfs to_csv with header sequentially
    """
    comm = MPI.COMM_WORLD
    hdfs_fname = hdfs_datapath("test_df_bodo_header_seq.csv")

    def test_write(test_df):
        test_df.to_csv(hdfs_fname, index=False)

    bodo_write = bodo.jit(test_write)
    err = None
    if bodo.get_rank() == 0:
        try:
            bodo_write(test_df)
        except Exception as e:
            err = e
    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err


def test_hdfs_csv_write_header_1D(hdfs_datapath, test_df):
    """
    test hdfs to_csv with header in 1D distributed
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_header_1D.csv")

    def test_write(test_df):
        test_df.to_csv(hdfs_fname, index=False)

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False))


def test_hdfs_csv_write_header_1D_var(hdfs_datapath, test_df):
    """
    test hdfs to_csv with header in 1D var
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_header_1D_var.csv")

    def test_write(test_df):
        test_df.to_csv(hdfs_fname, index=False)

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False, True))


def test_hdfs_json_write_records_lines_seq(hdfs_datapath, test_df):
    """
    test hdfs to_json(orient="records", lines=True) sequentially
    """
    comm = MPI.COMM_WORLD
    hdfs_fname = hdfs_datapath("df_records_lines_seq.json")

    def test_write(test_df, hdfs_fname):
        test_df.to_json(hdfs_fname, orient="records", lines=True)

    bodo_write = bodo.jit(test_write)
    err = None
    if bodo.get_rank() == 0:
        try:
            bodo_write(test_df, hdfs_fname)
        except Exception as e:
            err = e
    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err


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
            dtype={"A": float, "B": "bool", "C": int},
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
            dtype={"A": float, "B": "bool", "C": int},
        )

    check_func(test_read, (hdfs_fname,), py_output=test_df)


def test_hdfs_csv_read_1D_var(hdfs_datapath, test_df):
    """
    read_csv
    test the csv file we just wrote in 1D Var
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_1D_var.csv")

    def test_read(hdfs_fname):
        return pd.read_csv(
            hdfs_fname,
            names=["A", "B", "C"],
            dtype={"A": float, "B": "bool", "C": int},
        )

    check_func(test_read, (hdfs_fname,), py_output=test_df)


def test_hdfs_csv_read_header_seq(hdfs_datapath, test_df):
    """
    read_csv with header and infer dtypes
    test the csv file we just wrote sequentially
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_header_seq.csv")

    def test_read():
        return pd.read_csv(hdfs_fname)

    check_func(test_read, (), py_output=test_df)


def test_hdfs_csv_read_header_1D(hdfs_datapath, test_df):
    """
    read_csv with header and infer dtypes
    test the csv file we just wrote in 1D
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_header_1D.csv")

    def test_read():
        return pd.read_csv(hdfs_fname)

    check_func(test_read, (), py_output=test_df)


def test_hdfs_csv_read_1D_header_var(hdfs_datapath, test_df):
    """
    read_csv with header and infer dtypes
    test the csv file we just wrote in 1D Var
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_header_1D_var.csv")

    def test_read():
        return pd.read_csv(hdfs_fname)

    check_func(test_read, (), py_output=test_df)


@pytest.fixture(params=[np.arange(5)])
def test_np_arr(request):
    return request.param


def test_hdfs_np_tofile_seq(hdfs_datapath, test_np_arr):
    """
    test hdfs to_file
    """
    comm = MPI.COMM_WORLD

    def test_write(test_np_arr, hdfs_fname):
        test_np_arr.tofile(hdfs_fname)

    hdfs_fname = hdfs_datapath("test_np_arr_bodo_seq.dat")
    bodo_func = bodo.jit(test_write)
    err = None

    if bodo.get_rank() == 0:
        try:
            bodo_func(test_np_arr, hdfs_fname)
        except Exception as e:
            err = e
    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err


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


def test_hdfs_np_fromfile_seq_count_offset(hdfs_datapath, test_np_arr):
    """
    fromfile with count and offset
    """

    count = 2
    offset = 1

    def test_read(hdfs_fname):
        bytes_per_int64 = 8
        return np.fromfile(
            hdfs_fname, np.int64, count=count, offset=offset * bytes_per_int64
        )

    hdfs_fname = hdfs_datapath("test_np_arr_bodo_seq.dat")
    check_func(test_read, (hdfs_fname,), py_output=test_np_arr[offset : offset + count])


def test_hdfs_np_fromfile_seq_large_count(hdfs_datapath, test_np_arr):
    """
    fromfile with count larger than the length of the data
    test to read all the data and not throw an error
    """

    count = len(test_np_arr) + 1

    def test_read(hdfs_fname):
        return np.fromfile(hdfs_fname, np.int64, count=count)

    hdfs_fname = hdfs_datapath("test_np_arr_bodo_seq.dat")
    check_func(test_read, (hdfs_fname,), py_output=test_np_arr[:count])


@pytest.mark.skipif(bodo.get_size() > 1, reason="Observing errors in hadoop fs")
def test_hdfs_np_fromfile_seq_large_offset(hdfs_datapath, test_np_arr):
    """
    fromfile with offset larger than the length of the data
    this setup raises a ValueError which is expected
    """
    comm = MPI.COMM_WORLD

    offset = len(test_np_arr) + 1

    def test_read(hdfs_fname):
        bytes_per_int64 = 8
        return np.fromfile(hdfs_fname, np.int64, offset=offset * bytes_per_int64)

    hdfs_fname = hdfs_datapath("test_np_arr_bodo_seq.dat")
    err = None
    if bodo.get_rank() == 0:
        try:
            with pytest.raises(ValueError, match="negative dimensions not allowed"):
                bodo.jit(test_read)(hdfs_fname)
        except Exception as e:
            err = e
    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err


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
        )

    def test_read_infer_dtype(hdfs_fname):
        return pd.read_json(
            hdfs_fname,
            orient="records",
            lines=True,
            dtype={"A": float, "B": "bool", "C": int},
        )

    hdfs_fname = hdfs_datapath("df_records_lines_seq.json")
    check_func(test_read, (hdfs_fname,), py_output=test_df)
    check_func(test_read_infer_dtype, (hdfs_fname,), py_output=test_df)


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
        )

    def test_read_infer_dtype(hdfs_fname):
        return pd.read_json(
            hdfs_fname,
            orient="records",
            lines=True,
            dtype={"A": float, "B": "bool", "C": int},
        )

    hdfs_fname = hdfs_datapath("df_records_lines_1D.json")
    check_func(test_read, (hdfs_fname,), py_output=test_df)
    check_func(test_read_infer_dtype, (hdfs_fname,), py_output=test_df)


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
        )

    def test_read_infer_dtype(hdfs_fname):
        return pd.read_json(
            hdfs_fname,
            orient="records",
            lines=True,
            dtype={"A": float, "B": "bool", "C": int},
        )

    hdfs_fname = hdfs_datapath("df_records_lines_1D_var.json")
    check_func(test_read, (hdfs_fname,), py_output=test_df)
    check_func(test_read_infer_dtype, (hdfs_fname,), py_output=test_df)
