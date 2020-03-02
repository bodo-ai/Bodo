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
