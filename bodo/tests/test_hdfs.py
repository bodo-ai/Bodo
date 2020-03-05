# Copyright (C) 2019 Bodo Inc.
import pytest
import os
import pandas as pd
import numpy as np
import numba
import bodo
from bodo.utils.typing import BodoError
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

    try:
        bodo.io.parquet_pio.use_nullable_int_arr = True
        check_func(test_impl, (hdfs_fname,), py_output=py_output, check_dtype=False)
    finally:
        bodo.io.parquet_pio.use_nullable_int_arr = False


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


def test_hdfs_parquet_write_seq(datapath, hdfs_datapath, test_df):
    """
    test hdfs to_parquet sequentially
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_seq.pq")

    def test_write(test_df, hdfs_fname):
        test_df.to_parquet(hdfs_fname)

    bodo_write = bodo.jit(test_write)
    bodo_write(test_df, hdfs_fname)


def test_hdfs_parquet_write_1D(datapath, hdfs_datapath, test_df):
    """
    test hdfs to_parquet in 1D distributed
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_1D.pq")

    def test_write(test_df, hdfs_fname):
        test_df.to_parquet(hdfs_fname)

    bodo_write = bodo.jit(all_args_distributed=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False), hdfs_fname)


def test_hdfs_parquet_write_1D_var(datapath, hdfs_datapath, test_df):
    """
    test hdfs to_parquet in 1D Var distributed
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_1D_var.pq")

    def test_write(test_df, hdfs_fname):
        test_df.to_parquet(hdfs_fname)

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False, True), hdfs_fname)


def test_hdfs_parquet_read_seq(datapath, hdfs_datapath, test_df):
    """
    read_parquet sequentially
    test the parquet file we just wrote
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_seq.pq")

    def test_read(hdfs_fname):
        return pd.read_parquet(hdfs_fname)

    n_pes = bodo.get_size()
    bodo_read = bodo.jit(test_read)
    bodo_out = bodo_read(hdfs_fname)
    passed = _test_equal_guard(bodo_out, test_df, False)
    n_passed = reduce_sum(passed)
    assert n_passed == n_pes


def test_hdfs_parquet_read_1D(datapath, hdfs_datapath, test_df):
    """
    read_parquet in 1D
    test the parquet file we just wrote
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_1D.pq")

    def test_read(hdfs_fname):
        return pd.read_parquet(hdfs_fname)

    n_pes = bodo.get_size()
    bodo_read = bodo.jit(all_returns_distributed=True)(test_read)
    bodo_out = bodo_read(hdfs_fname)
    bodo_out = bodo.gatherv(bodo_out)
    passed = 1
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(bodo_out, test_df, False)
    n_passed = reduce_sum(passed)
    assert n_passed == n_pes


def test_hdfs_parquet_read_1D_var(datapath, hdfs_datapath, test_df):
    """
    read_parquet in 1D Var
    test the parquet file we just wrote
    """

    hdfs_fname = hdfs_datapath("test_df_bodo_1D_var.pq")

    def test_read(hdfs_fname):
        return pd.read_parquet(hdfs_fname)

    n_pes = bodo.get_size()
    bodo_read = bodo.jit(all_returns_distributed=True)(test_read)
    bodo_out = bodo_read(hdfs_fname)
    bodo_out = bodo.gatherv(bodo_out)
    passed = 1
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(bodo_out, test_df, False)
    n_passed = reduce_sum(passed)
    assert n_passed == n_pes
