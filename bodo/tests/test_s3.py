# Copyright (C) 2019 Bodo Inc.
import pytest
import pandas as pd
import numpy as np
import bodo
from bodo.tests.utils import check_func, _get_dist_arg, _test_equal_guard, reduce_sum
from bodo.utils.testing import ensure_clean

pytestmark = pytest.mark.s3


def test_s3_csv_data1(minio_server, s3_bucket, datapath):
    """
    test s3 read_csv
    """

    def test_impl():
        return pd.read_csv(
            "s3://bodo-test/csv_data1.csv",
            names=["A", "B", "C", "D"],
            dtype={"A": np.int, "B": np.float, "C": np.float, "D": np.int},
        )

    fname = datapath("csv_data1.csv")
    py_output = pd.read_csv(
        fname,
        names=["A", "B", "C", "D"],
        dtype={"A": np.int, "B": np.float, "C": np.float, "D": np.int},
    )

    check_func(test_impl, (), py_output=py_output)


def test_s3_csv_data1_compressed(minio_server, s3_bucket, datapath):
    """
    test s3 read_csv
    """

    def test_impl_gzip():
        return pd.read_csv("s3://bodo-test/csv_data1.csv.gz",
                           names=["A", "B", "C", "D"],
                           header=None)

    def test_impl_bz2():
        return pd.read_csv("s3://bodo-test/csv_data1.csv.bz2",
                           names=["A", "B", "C", "D"],
                           header=None)

    fname = datapath("csv_data1.csv")
    py_output = pd.read_csv(fname, names=["A", "B", "C", "D"], header=None)

    check_func(test_impl_gzip, (), py_output=py_output)
    check_func(test_impl_bz2, (), py_output=py_output)


def test_s3_csv_data_date1(minio_server, s3_bucket, datapath):
    """
    test s3 read_csv
    """

    def test_impl():
        return pd.read_csv(
            "s3://bodo-test/csv_data_date1.csv",
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
    check_func(test_impl, (), py_output=py_output)


def test_s3_pq_asof1(minio_server, s3_bucket, datapath):
    """
    test s3 read_parquet
    """

    def test_impl():
        return pd.read_parquet("s3://bodo-test/asof1.pq")

    fname = datapath("asof1.pq")
    py_output = pd.read_parquet(fname)
    check_func(test_impl, (), py_output=py_output)


def test_s3_pq_groupby3(minio_server, s3_bucket, datapath):
    """
    test s3 read_parquet
    """

    def test_impl():
        return pd.read_parquet("s3://bodo-test/groupby3.pq")

    fname = datapath("groupby3.pq")
    py_output = pd.read_parquet(fname)
    check_func(test_impl, (), py_output=py_output)


def test_s3_read_json(minio_server, s3_bucket, datapath):
    """
    test read_json from s3
    """
    fname_file = "s3://bodo-test/example.json"
    fname_dir_single = "s3://bodo-test/example_single.json"
    fname_dir_multi = "s3://bodo-test/example_multi.json"

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


def test_s3_parquet_write_seq(minio_server, s3_bucket, test_df):
    """
    test s3 to_parquet sequentially
    """

    def test_write(test_df):
        test_df.to_parquet("s3://bodo-test/test_df_bodo_seq.pq")

    bodo_write = bodo.jit(test_write)
    bodo_write(test_df)


def test_s3_parquet_write_1D(minio_server, s3_bucket, test_df):
    """
    test s3 to_parquet in 1D distributed
    """

    def test_write(test_df):
        test_df.to_parquet("s3://bodo-test/test_df_bodo_1D.pq")

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False))


def test_s3_parquet_write_1D_var(minio_server, s3_bucket, test_df):
    """
    test s3 to_parquet in 1D var
    """

    def test_write(test_df):
        test_df.to_parquet("s3://bodo-test/test_df_bodo_1D_var.pq")

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False, True))


def test_s3_csv_write_seq(minio_server, s3_bucket, test_df):
    """
    test s3 to_csv sequentially
    """

    def test_write(test_df):
        test_df.to_csv("s3://bodo-test/test_df_bodo_seq.csv", index=False, header=False)

    bodo_write = bodo.jit(test_write)
    bodo_write(test_df)


def test_s3_csv_write_1D(minio_server, s3_bucket, test_df):
    """
    test s3 to_csv in 1D distributed
    """

    def test_write(test_df):
        test_df.to_csv("s3://bodo-test/test_df_bodo_1D.csv", index=False, header=False)

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False))


def test_s3_csv_write_1D_var(minio_server, s3_bucket, test_df):
    """
    test s3 to_csv in 1D var
    """

    def test_write(test_df):
        test_df.to_csv(
            "s3://bodo-test/test_df_bodo_1D_var.csv", index=False, header=False
        )

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False, True))


def test_s3_csv_write_header_seq(minio_server, s3_bucket, test_df):
    """
    test s3 to_csv with header sequentially
    """

    def test_write(test_df):
        test_df.to_csv("s3://bodo-test/test_df_bodo_header_seq.csv", index=False)

    bodo_write = bodo.jit(test_write)
    bodo_write(test_df)


def test_s3_csv_write_header_1D(minio_server, s3_bucket, test_df):
    """
    test s3 to_csv with header in 1D distributed
    """

    def test_write(test_df):
        test_df.to_csv("s3://bodo-test/test_df_bodo_header_1D.csv", index=False)

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False))


def test_s3_csv_write_header_1D_var(minio_server, s3_bucket, test_df):
    """
    test s3 to_csv with header in 1D var
    """

    def test_write(test_df):
        test_df.to_csv("s3://bodo-test/test_df_bodo_header_1D_var.csv", index=False)

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False, True))


def test_s3_json_write_records_lines_seq(minio_server, s3_bucket, test_df):
    """
    test s3 to_json(orient="records", lines=True) sequentially
    """

    def test_write(test_df):
        test_df.to_json(
            "s3://bodo-test/df_records_lines_seq.json", orient="records", lines=True
        )

    bodo_write = bodo.jit(test_write)
    bodo_write(test_df)


def test_s3_json_write_records_lines_1D(minio_server, s3_bucket, test_df):
    """
    test s3 to_json(orient="records", lines=True) in 1D distributed
    """

    def test_write(test_df):
        test_df.to_json(
            "s3://bodo-test/df_records_lines_1D.json", orient="records", lines=True
        )

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False))


def test_s3_json_write_records_lines_1D_var(minio_server, s3_bucket, test_df):
    """
    test s3 to_json(orient="records", lines=True) in 1D var
    """

    def test_write(test_df):
        test_df.to_json(
            "s3://bodo-test/df_records_lines_1D_var.json", orient="records", lines=True
        )

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False, True))


def test_s3_parquet_read_seq(minio_server, s3_bucket, test_df):
    """
    read_parquet
    test the parquet file we just wrote sequentially
    """

    def test_read():
        return pd.read_parquet("s3://bodo-test/test_df_bodo_seq.pq")

    check_func(test_read, (), py_output=test_df)


def test_s3_parquet_read_1D(minio_server, s3_bucket, test_df, datapath):
    """
    read_parquet
    test the parquet file we just wrote in 1D
    """

    def test_read():
        return pd.read_parquet("s3://bodo-test/test_df_bodo_1D.pq")

    check_func(test_read, (), py_output=test_df)


def test_s3_parquet_read_1D_var(minio_server, s3_bucket, test_df):
    """
    read_parquet
    test the parquet file we just wrote  in 1D Var
    """

    def test_read():
        return pd.read_parquet("s3://bodo-test/test_df_bodo_1D_var.pq")

    check_func(test_read, (), py_output=test_df)


def test_s3_csv_read_seq(minio_server, s3_bucket, test_df):
    """
    read_csv 
    test the csv file we just wrote sequentially
    """

    def test_read():
        return pd.read_csv(
            "s3://bodo-test/test_df_bodo_seq.csv",
            names=["A", "B", "C"],
            dtype={"A": np.float, "B": "bool", "C": np.int},
        )

    check_func(test_read, (), py_output=test_df)


def test_s3_csv_read_1D(minio_server, s3_bucket, test_df):
    """
    read_csv 
    test the csv file we just wrote in 1D
    """

    def test_read():
        return pd.read_csv(
            "s3://bodo-test/test_df_bodo_1D.csv",
            names=["A", "B", "C"],
            dtype={"A": np.float, "B": "bool", "C": np.int},
        )

    check_func(test_read, (), py_output=test_df)


def test_s3_csv_read_1D_var(minio_server, s3_bucket, test_df):
    """
    read_csv 
    test the csv file we just wrote in 1D Var
    """

    def test_read():
        return pd.read_csv(
            "s3://bodo-test/test_df_bodo_1D_var.csv",
            names=["A", "B", "C"],
            dtype={"A": np.float, "B": "bool", "C": np.int},
        )

    check_func(test_read, (), py_output=test_df)


def test_s3_csv_read_header_seq(minio_server, s3_bucket, test_df):
    """
    read_csv with header and infer dtypes
    test the csv file we just wrote sequentially
    """

    def test_read():
        return pd.read_csv("s3://bodo-test/test_df_bodo_header_seq.csv",)

    check_func(test_read, (), py_output=test_df)


def test_s3_csv_read_header_1D(minio_server, s3_bucket, test_df):
    """
    read_csv with header and infer dtypes
    test the csv file we just wrote in 1D
    """

    def test_read():
        return pd.read_csv("s3://bodo-test/test_df_bodo_header_1D.csv",)

    check_func(test_read, (), py_output=test_df)


def test_s3_csv_read_1D_header_var(minio_server, s3_bucket, test_df):
    """
    read_csv with header and infer dtypes
    test the csv file we just wrote in 1D Var
    """

    def test_read():
        return pd.read_csv("s3://bodo-test/test_df_bodo_header_1D_var.csv",)

    check_func(test_read, (), py_output=test_df)


@pytest.fixture(params=[np.arange(5)])
def test_np_arr(request):
    return request.param


def test_s3_np_tofile_seq(minio_server, s3_bucket, test_np_arr):
    """
    test s3 to_file
    """

    def test_write(test_np_arr):
        test_np_arr.tofile("s3://bodo-test/test_np_arr_bodo_seq.dat")

    bodo_write = bodo.jit(test_write)
    bodo_write(test_np_arr)


def test_s3_np_tofile_1D(minio_server, s3_bucket, test_np_arr):
    """
    test s3 to_file in 1D distributed
    """

    def test_write(test_np_arr):
        test_np_arr.tofile("s3://bodo-test/test_np_arr_bodo_1D.dat")

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_np_arr, True))


def test_s3_np_tofile_1D_var(minio_server, s3_bucket, test_np_arr):
    """
    test s3 to_file in 1D Var
    """

    def test_write(test_np_arr):
        test_np_arr.tofile("s3://bodo-test/test_np_arr_bodo_1D_var.dat")

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_np_arr, True, True))


def test_s3_np_fromfile_seq(minio_server, s3_bucket, test_np_arr):
    """
    fromfile
    test the dat file we just wrote sequentially
    """

    def test_read():
        return np.fromfile("s3://bodo-test/test_np_arr_bodo_seq.dat", np.int64)

    bodo_func = bodo.jit(test_read)
    check_func(test_read, (), py_output=test_np_arr)


def test_s3_np_fromfile_1D(minio_server, s3_bucket, test_np_arr):
    """
    fromfile
    test the dat file we just wrote in 1D
    """

    def test_read():
        return np.fromfile("s3://bodo-test/test_np_arr_bodo_1D.dat", np.int64)

    bodo_func = bodo.jit(test_read)
    check_func(test_read, (), py_output=test_np_arr)


def test_s3_np_fromfile_1D_var(minio_server, s3_bucket, test_np_arr):
    """
    fromfile
    test the dat file we just wrote in 1D
    """

    def test_read():
        return np.fromfile("s3://bodo-test/test_np_arr_bodo_1D_var.dat", np.int64)

    bodo_func = bodo.jit(test_read)
    check_func(test_read, (), py_output=test_np_arr)


def test_s3_json_read_records_lines_seq(minio_server, s3_bucket, test_df):
    """
    read_json(orient="records", lines=True)
    test the json file we just wrote sequentially
    """

    def test_read():
        return pd.read_json(
            "s3://bodo-test/df_records_lines_seq.json", orient="records", lines=True,
        )

    def test_read_infer_dtype():
        return pd.read_json(
            "s3://bodo-test/df_records_lines_seq.json",
            orient="records",
            lines=True,
            dtype={"A": np.float, "B": "bool", "C": np.int},
        )

    check_func(test_read, (), py_output=test_df)
    check_func(test_read_infer_dtype, (), py_output=test_df)


def test_s3_json_read_records_lines_1D(minio_server, s3_bucket, test_df):
    """
    read_json(orient="records", lines=True) 
    test the json file we just wrote in 1D
    """

    def test_read():
        return pd.read_json(
            "s3://bodo-test/df_records_lines_1D.json", orient="records", lines=True,
        )

    def test_read_infer_dtype():
        return pd.read_json(
            "s3://bodo-test/df_records_lines_1D.json", orient="records", lines=True,
        )

    check_func(test_read, (), py_output=test_df)
    check_func(test_read_infer_dtype, (), py_output=test_df)


def test_s3_json_read_recoreds_lines_1D_var(minio_server, s3_bucket, test_df):
    """
    read_json(orient="records", lines=True) 
    test the json file we just wrote in 1D Var
    """

    def test_read():
        return pd.read_json(
            "s3://bodo-test/df_records_lines_1D_var.json", orient="records", lines=True,
        )

    def test_read_infer_dtype():
        return pd.read_json(
            "s3://bodo-test/df_records_lines_1D_var.json",
            orient="records",
            lines=True,
            dtype={"A": np.float, "B": "bool", "C": np.int},
        )

    check_func(test_read, (), py_output=test_df)
    check_func(test_read_infer_dtype, (), py_output=test_df)
