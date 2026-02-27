import os
import tempfile

import pandas as pd
import pytest

import bodo.pandas as bd
from bodo.tests.utils import _test_equal


def test_gpu_join(datapath):
    """Test end-to-end Read-Join-Write workflow on GPU."""
    cust_path = datapath("tpch-test_data/parquet/customer.pq")
    orders_path = datapath("tpch-test_data/parquet/orders.pq")

    def merge_impl(cust_df, orders_df, out_path):
        merged = cust_df.merge(
            orders_df, how="inner", left_on=["C_CUSTKEY"], right_on=["O_CUSTKEY"]
        )
        merged.to_parquet(out_path)

    with tempfile.TemporaryDirectory() as tmp:
        # TODO: Check that entire pipeline is being executed on GPU.
        cust_bodo = bd.read_parquet(cust_path)
        orders_bodo = bd.read_parquet(orders_path)
        out_path_bodo = os.path.join(tmp, "out_bodo.pq")
        merge_impl(cust_bodo, orders_bodo, out_path_bodo)

        cust_pd = pd.read_parquet(cust_path)
        orders_pd = pd.read_parquet(orders_path)
        out_path_pd = os.path.join(tmp, "out_pd.pq")
        merge_impl(cust_pd, orders_pd, out_path_pd)

        result_bodo = pd.read_parquet(out_path_bodo)
        result_pd = pd.read_parquet(out_path_pd)
        _test_equal(result_bodo, result_pd, sort_output=True, reset_index=True)


def test_project_filter1(datapath):
    """Test end-to-end projection and filter workflow on GPU."""
    df1_path = datapath("dataframe_library/df1.parquet")

    def merge_impl(df1_df, out_path):
        proj_df = df1_df[["B", "D"]]
        filt_df = proj_df[proj_df.D > 30]
        filt_df.to_parquet(out_path)

    with tempfile.TemporaryDirectory() as tmp:
        # TODO: Check that entire pipeline is being executed on GPU.
        df1_bodo = bd.read_parquet(df1_path)
        out_path_bodo = os.path.join(tmp, "out_bodo.pq")
        merge_impl(df1_bodo, out_path_bodo)

        df1_pd = pd.read_parquet(df1_path)
        out_path_pd = os.path.join(tmp, "out_pd.pq")
        merge_impl(df1_pd, out_path_pd)

        result_bodo = pd.read_parquet(out_path_bodo)
        result_pd = pd.read_parquet(out_path_pd)
        _test_equal(result_bodo, result_pd, sort_output=True, reset_index=True)


@pytest.mark.parametrize(
    "func",
    [
        "sum",
        "mean",
        "count",
        "max",
        "min",
        "nunique",
        "size",
        "var",
        "std",
        "skew",
        "distinct",
    ],
)
def test_groupby_agg(datapath, func):
    """Test end-to-end projection and filter workflow on GPU."""
    df1_path = datapath("dataframe_library/df1.parquet")

    def merge_impl(df1_df, out_path):
        if func == "distinct":
            df1_agg = df1_df.drop_duplicates()
        else:
            df1_agg = getattr(
                df1_df.groupby("D", as_index=False, sort=False)["A"], func
            )()

        df1_agg.to_parquet(out_path)

    with tempfile.TemporaryDirectory() as tmp:
        # TODO: Check that entire pipeline is being executed on GPU.
        df1_bodo = bd.read_parquet(df1_path)
        out_path_bodo = os.path.join(tmp, "out_bodo.pq")
        merge_impl(df1_bodo, out_path_bodo)

        df1_pd = pd.read_parquet(df1_path)
        out_path_pd = os.path.join(tmp, "out_pd.pq")
        merge_impl(df1_pd, out_path_pd)

        result_bodo = pd.read_parquet(out_path_bodo)
        result_pd = pd.read_parquet(out_path_pd)
        _test_equal(result_bodo, result_pd, sort_output=True, reset_index=True)
