# Copyright (C) 2019 Bodo Inc. All rights reserved.
import pandas as pd
import pytest

from bodo.tests.utils import check_caching
from caching_tests_common import fn_distribution, is_cached


@pytest.mark.smoke
def test_read_csv_cache(fn_distribution, is_cached, datapath):
    # TODO: investigate/fix memory leak check, see BE-1375
    """
    test read_csv with cache=True
    """
    fname = datapath("csv_data1.csv")

    def impl():
        df = pd.read_csv(fname, names=["A", "B", "C", "D"], compression=None)
        return df.C

    check_caching(impl, (), is_cached, fn_distribution)


@pytest.mark.smoke
def test_read_parquet_cache(fn_distribution, is_cached, datapath):
    # TODO: investigate/fix memory leak check, see BE-1375
    """
    test read_parquet with cache=True
    """

    def impl(fname):
        return pd.read_parquet(fname)

    fname = datapath("groupby3.pq")
    check_caching(impl, (fname,), is_cached, fn_distribution)


def test_read_parquet_cache_fname_arg(
    fn_distribution, is_cached, datapath, memory_leak_check
):
    """
    test read_parquet with cache=True and passing different file name as
    argument to the Bodo function
    """

    def impl(fname):
        return pd.read_parquet(fname)

    def impl2(fname):
        return pd.read_parquet(fname)

    fname1 = datapath("int_nulls_single.pq")
    fname2 = datapath("int_nulls_multi.pq")

    py_out = impl(fname1)
    check_caching(impl, (fname1,), is_cached, fn_distribution, py_output=py_out)
    check_caching(impl2, (fname2,), is_cached, fn_distribution, py_output=py_out)


def test_read_csv_cache_fname_arg(fn_distribution, is_cached, datapath):
    # TODO: investigate/fix memory leak check, see BE-1375
    """
    test read_csv with cache=True and passing different file name as
    argument to the Bodo function
    """

    def impl(fname):
        return pd.read_csv(fname)

    def impl2(fname):
        return pd.read_csv(fname)

    fname1 = datapath("example.csv")
    fname2 = datapath("example_multi.csv")  # directory of csv files

    py_out = impl(fname1)
    check_caching(impl, (fname1,), is_cached, fn_distribution, py_output=py_out)
    check_caching(impl2, (fname2,), is_cached, fn_distribution, py_output=py_out)


def test_read_json_cache_fname_arg(
    fn_distribution, is_cached, datapath, memory_leak_check
):
    """
    test read_json with cache=True and passing different file name as
    argument to the Bodo function
    """

    def impl(fname):
        return pd.read_json(fname, orient="records", lines=True)

    def impl2(fname):
        return pd.read_json(fname, orient="records", lines=True)

    fname1 = datapath("example.json")
    fname2 = datapath("example_single.json")  # directory with one json file

    py_out = impl(fname1)
    check_caching(impl, (fname1,), is_cached, fn_distribution, py_output=py_out)
    check_caching(impl2, (fname2,), is_cached, fn_distribution, py_output=py_out)
