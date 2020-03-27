# Copyright (C) 2019 Bodo Inc.
import pytest
import pandas as pd
import numpy as np
import bodo
from bodo.tests.utils import check_func, _get_dist_arg, _test_equal_guard, reduce_sum

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

    bodo_output = bodo.jit(test_impl)()
    check_func(test_impl, (), py_output=py_output)


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

    bodo_write = bodo.jit(all_args_distributed=True)(test_write)
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
        test_df.to_csv("s3://bodo-test/test_df_bodo_seq.csv")

    bodo_write = bodo.jit(test_write)
    bodo_write(test_df)


def test_s3_csv_write_1D(minio_server, s3_bucket, test_df):
    """
    test s3 to_csv in 1D distributed
    """

    def test_write(test_df):
        test_df.to_csv("s3://bodo-test/test_df_bodo_1D.csv")

    bodo_write = bodo.jit(all_args_distributed=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False))


def test_s3_csv_write_1D_var(minio_server, s3_bucket, test_df):
    """
    test s3 to_csv in 1D var
    """

    def test_write(test_df):
        test_df.to_csv("s3://bodo-test/test_df_bodo_1D_var.csv")

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False, True))


def test_s3_parquet_read_seq(minio_server, s3_bucket, test_df):
    """
    read_parquet sequentially
    test the parquet file we just wrote
    """

    def test_read():
        return pd.read_parquet("s3://bodo-test/test_df_bodo_seq.pq")

    n_pes = bodo.get_size()
    bodo_read = bodo.jit(test_read)
    bodo_out = bodo_read()
    passed = _test_equal_guard(bodo_out, test_df, False)
    n_passed = reduce_sum(passed)
    assert n_passed == n_pes


def test_s3_parquet_read_1D(minio_server, s3_bucket, test_df, datapath):
    """
    read_parquet in 1D
    test the parquet file we just wrote
    """

    def test_read():
        return pd.read_parquet("s3://bodo-test/test_df_bodo_1D.pq")

    n_pes = bodo.get_size()
    bodo_read = bodo.jit(all_returns_distributed=True)(test_read)
    bodo_out = bodo_read()
    bodo_out = bodo.gatherv(bodo_out)
    passed = 1
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(bodo_out, test_df, False)
    n_passed = reduce_sum(passed)
    assert n_passed == n_pes


def test_s3_parquet_read_1D_var(minio_server, s3_bucket, test_df):
    """
    read_parquet in 1D Var
    test the parquet file we just wrote
    """

    def test_read():
        return pd.read_parquet("s3://bodo-test/test_df_bodo_1D_var.pq")

    n_pes = bodo.get_size()
    bodo_read = bodo.jit(all_returns_distributed=True)(test_read)
    bodo_out = bodo_read()
    bodo_out = bodo.gatherv(bodo_out)
    passed = 1
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(bodo_out, test_df, False)
    n_passed = reduce_sum(passed)
    assert n_passed == n_pes
