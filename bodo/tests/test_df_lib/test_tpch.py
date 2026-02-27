import numpy as np
import pandas as pd
import pytest

import benchmarks.tpch.bodo.dataframe_queries as tpch
import bodo.pandas as bd
from bodo.pandas.plan import assert_executed_plan_count
from bodo.tests.utils import _test_equal

pytestmark = pytest.mark.jit_dependency

datapath = "bodo/tests/data/tpch-test_data/parquet"


def run_tpch_query_test(query_func, plan_executions=0, ctes_created=0):
    """Run a tpch query and compare output to Pandas.

    Args:
        query_func (Callable): The callable object that takes in dataframes loaded from
          TPCH and returns an output dataframe.
        plan_executions (int, optional): Expected number of LazyPlans to be executed.
          Defaults to 0.
    """

    # Scale factor is set to 1.0 for testing purposes in query 11
    pd_kwargs = {"pd": pd}
    pd_args = [
        getattr(tpch, f"load_{key}")(datapath, **pd_kwargs)
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
    bd_kwargs = {"pd": bd}
    bd_args = [
        getattr(tpch, f"load_{key}")(datapath, **bd_kwargs)
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

    pd_result = query_func(*pd_args, **pd_kwargs)

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


def test_tpch_q01():
    run_tpch_query_test(tpch.tpch_q01)


def test_tpch_q02():
    run_tpch_query_test(tpch.tpch_q02, ctes_created=1)


def test_tpch_q03():
    run_tpch_query_test(tpch.tpch_q03)


def test_tpch_q04():
    run_tpch_query_test(tpch.tpch_q04)


def test_tpch_q05():
    run_tpch_query_test(tpch.tpch_q05)


def test_tpch_q06():
    run_tpch_query_test(tpch.tpch_q06, plan_executions=1)


def test_tpch_q07():
    run_tpch_query_test(tpch.tpch_q07)


def test_tpch_q08():
    run_tpch_query_test(tpch.tpch_q08, ctes_created=1)


def test_tpch_q09():
    run_tpch_query_test(tpch.tpch_q09)


def test_tpch_q10():
    run_tpch_query_test(tpch.tpch_q10)


def test_tpch_q11():
    run_tpch_query_test(tpch.tpch_q11, ctes_created=1)


def test_tpch_q12():
    run_tpch_query_test(tpch.tpch_q12)


def test_tpch_q13():
    run_tpch_query_test(tpch.tpch_q13)


def test_tpch_q14():
    run_tpch_query_test(tpch.tpch_q14, plan_executions=1)


def test_tpch_q15():
    run_tpch_query_test(tpch.tpch_q15, ctes_created=1)


def test_tpch_q16():
    run_tpch_query_test(tpch.tpch_q16)


def test_tpch_q17():
    run_tpch_query_test(tpch.tpch_q17, plan_executions=1)


def test_tpch_q18():
    run_tpch_query_test(tpch.tpch_q18)


def test_tpch_q19():
    run_tpch_query_test(tpch.tpch_q19, plan_executions=1)


def test_tpch_q20():
    run_tpch_query_test(tpch.tpch_q20)


def test_tpch_q21():
    run_tpch_query_test(tpch.tpch_q21, ctes_created=1)


def test_tpch_q22():
    run_tpch_query_test(tpch.tpch_q22, ctes_created=1)
