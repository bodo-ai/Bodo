import numpy as np
import pandas as pd
import pytest

import benchmarks.tpch.dataframe_lib as tpch
import bodo.pandas as bd
from bodo.pandas.plan import assert_executed_plan_count
from bodo.tests.utils import _test_equal

pytestmark = pytest.mark.jit_dependency

datapath = "bodo/tests/data/tpch-test_data/parquet"


def run_tpch_query_test(query_func, plan_executions=0):
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
        if key != "scale_factor"
    ]
    bd_args = [bd.from_pandas(df) for df in pd_args]

    pd_result = query_func(*pd_args, **pd_kwargs)

    with assert_executed_plan_count(plan_executions):
        bd_result = query_func(*bd_args)

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
    run_tpch_query_test(tpch.tpch_q02)


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
    run_tpch_query_test(tpch.tpch_q08)


def test_tpch_q09():
    run_tpch_query_test(tpch.tpch_q09)


def test_tpch_q10():
    run_tpch_query_test(tpch.tpch_q10)


def test_tpch_q11():
    run_tpch_query_test(tpch.tpch_q11, plan_executions=1)


def test_tpch_q12():
    run_tpch_query_test(tpch.tpch_q12)


def test_tpch_q13():
    run_tpch_query_test(tpch.tpch_q13)


def test_tpch_q14():
    # TODO [BSE-5099]: Series.where
    run_tpch_query_test(tpch.tpch_q14, plan_executions=5)


def test_tpch_q15():
    run_tpch_query_test(tpch.tpch_q15)


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
    run_tpch_query_test(tpch.tpch_q21)


def test_tpch_q22():
    run_tpch_query_test(
        tpch.tpch_q22,
    )
