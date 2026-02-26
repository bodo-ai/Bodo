import os
import tempfile

import pandas as pd

import bodo.pandas as bd
from bodo.pandas.base import _empty_like
from bodo.pandas.plan import (
    LazyPlan,
    LogicalParquetWrite,
    count_gpu_plan_nodes,
    getPlanStatistics,
)
from bodo.tests.utils import _test_equal


def is_gpu_plan(plan: LazyPlan) -> bool:
    """Check whether entire plan is being executed on GPU."""
    num_gpu_nodes = count_gpu_plan_nodes(plan)
    _, total_optimized_nodes = getPlanStatistics(plan)
    return num_gpu_nodes == total_optimized_nodes


def test_gpu_join(datapath):
    """Test end-to-end Read-Join-Write workflow on GPU."""
    cust_path = datapath("tpch-test_data/parquet/customer.pq")
    orders_path = datapath("tpch-test_data/parquet/orders.pq")

    def merge_impl(cust_df, orders_df, out_path):
        merged = cust_df.merge(
            orders_df, how="inner", left_on=["C_CUSTKEY"], right_on=["O_CUSTKEY"]
        )
        if isinstance(merged, bd.DataFrame):
            assert is_gpu_plan(merged._plan), "Expected entire plan to run on GPU"
        merged.to_parquet(out_path)

    with tempfile.TemporaryDirectory() as tmp:
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


def test_cpu_to_gpu_exchange(datapath):
    """Test pipelines that transfers data between CPU and GPU"""

    path = datapath("dataframe_library/df1.parquet")

    pdf1 = pd.DataFrame({"A": range(30)})
    bdf1 = bd.from_pandas(pdf1)
    bdf2 = bd.read_parquet(path)
    # Case 1: CPU (read pandas) -> GPU process batch (join probe)
    bdf3 = bdf1.merge(bdf2, how="inner", on="A")
    # ReadParquet, JoinFilter, Join, Project are all on GPU
    # Read DF should be on CPU
    assert count_gpu_plan_nodes(bdf3._plan) == 4
    bdf3 = bdf3.sort_values(bdf3.columns.to_list())

    pdf2 = pd.read_parquet(path)
    pdf3 = pdf1.merge(pdf2, how="inner", on="A")
    _test_equal(pdf3, bdf3)

    # Case 2: CPU (read pandas) -> GPU sink (write parquet)
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "out_bodo.pq")
        pdf = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bdf = bd.from_pandas(pdf)
        bdf.to_parquet(out_path)

        out_df = pd.read_parquet(out_path)
        # [BSE-5322] Sorting data because CPU -> GPU can change order
        _test_equal(pdf, out_df, sort_output=True, reset_index=True)

        # Check the write parquet is happening on GPU:
        write_plan = LogicalParquetWrite(
            _empty_like(bdf),
            bdf._plan,
            out_path,
            "none",  # compression
            "",  # bucket_region
            -1,  # row_group_size
        )
        assert count_gpu_plan_nodes(write_plan) == 1


def test_gpu_to_cpu_exchange(datapath):
    """Test pipelines that transfers data between GPU and CPU"""
    pass
