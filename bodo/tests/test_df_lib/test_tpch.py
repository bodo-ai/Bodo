import numpy as np
import pandas as pd
import pytest
import s3fs

import benchmarks.tpch.bodo.dataframe_queries as tpch
import bodo.pandas as bd
from bodo.pandas.plan import assert_executed_plan_count
from bodo.tests.utils import _test_equal, set_broadcast_join

pytestmark = pytest.mark.jit_dependency

datapath = "s3://tpch-data-parquet/SF1"
expected_output_path = "bodo/tests/data/tpch-test_data/sf1_expected_output"


@pytest.fixture(scope="module")
def local_sf1_data(tmp_path_factory):
    local_sf1_dir = str(tmp_path_factory.mktemp("s3_sf1_data"))
    # Only copy needed Parquet files (exclude "output" and "outputs" folders)
    tpch_data_files = [
        "lineitem.pq",
        "part.pq",
        "orders.pq",
        "customer.pq",
        "nation.pq",
        "region.pq",
        "supplier.pq",
        "partsupp.pq",
    ]

    fs = s3fs.S3FileSystem()
    for file in tpch_data_files:
        fs.get(datapath + "/" + file, local_sf1_dir, recursive=True)

    return local_sf1_dir


def run_tpch_query_test(sf1_path, query_func, plan_executions=0, ctes_created=0):
    """Run a tpch query and compare output to Pandas (pre-computed).

    Args:
        sf1_path (str): The directory containing the TPC-H test data (SF1).
        query_func (Callable): The callable object that takes in dataframes loaded from
          TPCH and returns an output dataframe.
        plan_executions (int, optional): Expected number of LazyPlans to be executed.
          Defaults to 0.
    """

    # Scale factor is set to 1.0 for testing purposes in query 11
    bd_kwargs = {"pd": bd}
    bd_args = [
        getattr(tpch, f"load_{key}")(sf1_path, **bd_kwargs)
        for key in tpch._query_to_args[int(query_func.__name__[-2:])]
        if key
        in [
            "lineitem",
            "part",
            "orders",
            "customer",
            "nation",
            "region",
            "supplier",
            "partsupp",
        ]
    ]

    pd_result = pd.read_parquet(
        f"{expected_output_path}/{query_func.__name__[-3:]:02}.pq",
        dtype_backend="pyarrow",
    )

    with assert_executed_plan_count(plan_executions):
        bd_result = query_func(*bd_args)

    # We can't capture all CTEs created because the above should create
    # and execute plans if plan_executions > 0.  If those plans executed
    # above have CTEs then we can't capture them since they may occur on
    # workers.
    generated_ctes = bd_result._plan.get_cte_count()
    assert generated_ctes == ctes_created

    if isinstance(
        pd_result,
        (
            pd.DataFrame,
            pd.Series,
        ),
    ):
        _test_equal(
            bd_result,
            pd_result,
            check_pandas_types=False,
            sort_output=True,
            reset_index=True,
        )
    else:
        # For scalar or numeric results
        assert np.isclose(pd_result, bd_result)


@pytest.mark.gpu
@pytest.mark.parametrize("broadcast", [True, False])
def test_tpch_q01(local_sf1_data, broadcast):
    with set_broadcast_join(broadcast):
        run_tpch_query_test(local_sf1_data, tpch.tpch_q01)


@pytest.mark.gpu
def test_tpch_q02(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q02, ctes_created=1)


@pytest.mark.gpu
def test_tpch_q03(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q03)


@pytest.mark.gpu
def test_tpch_q04(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q04)


@pytest.mark.gpu
@pytest.mark.parametrize("broadcast", [True, False])
def test_tpch_q05(local_sf1_data, broadcast):
    with set_broadcast_join(broadcast):
        run_tpch_query_test(local_sf1_data, tpch.tpch_q05)


@pytest.mark.gpu
def test_tpch_q06(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q06, plan_executions=1)


@pytest.mark.gpu
def test_tpch_q07(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q07)


@pytest.mark.gpu
def test_tpch_q08(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q08, ctes_created=1)


@pytest.mark.gpu
def test_tpch_q09(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q09)


@pytest.mark.gpu
def test_tpch_q10(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q10)


@pytest.mark.gpu
def test_tpch_q11(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q11, ctes_created=1)


@pytest.mark.gpu
def test_tpch_q12(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q12)


@pytest.mark.gpu
def test_tpch_q13(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q13)


@pytest.mark.gpu
def test_tpch_q14(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q14, plan_executions=1)


@pytest.mark.gpu
def test_tpch_q15(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q15, ctes_created=1)


@pytest.mark.gpu
def test_tpch_q16(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q16)


@pytest.mark.gpu
def test_tpch_q17(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q17, plan_executions=1)


@pytest.mark.gpu
def test_tpch_q18(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q18)


@pytest.mark.gpu
def test_tpch_q19(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q19, plan_executions=1)


@pytest.mark.gpu
def test_tpch_q20(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q20)


@pytest.mark.gpu
def test_tpch_q21(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q21, ctes_created=1)


@pytest.mark.gpu
def test_tpch_q22(local_sf1_data):
    run_tpch_query_test(local_sf1_data, tpch.tpch_q22, ctes_created=1)
