import os
import tempfile

import pandas as pd
import pytest

import bodo.pandas as bd
from bodo.tests.utils import _test_equal

pytestmark = [pytest.mark.gpu]


@pytest.mark.gpu
def test_gpu_join(datapath):
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
