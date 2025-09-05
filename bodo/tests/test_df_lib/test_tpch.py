import pandas as pd
import pytest

import benchmarks.tpch.dataframe_lib as tpch
import bodo.pandas as bd
from bodo.pandas.plan import assert_executed_plan_count
from bodo.tests.utils import _test_equal

datapath = "bodo/tests/data/tpch-test_data/parquet"


def run_tpch_query_test(query_func1, query_func2, plan_executions=0):
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
        for key in tpch._query_to_args[int(query_func2.__name__[-2:])]
        if key != "scale_factor"
    ]
    bd_args = [bd.from_pandas(df) for df in pd_args]

    ENABLE_PLAN_CHECKING = 1
    pd_result2 = query_func2(*pd_args, **pd_kwargs)
    pd_result1 = query_func1(*pd_args, **pd_kwargs)
    if ENABLE_PLAN_CHECKING:
        with assert_executed_plan_count(plan_executions):
            bd_result1 = query_func1(*bd_args)
        # with assert_executed_plan_count(plan_executions):
        #     bd_result2 = query_func2(*bd_args)
    else:
        bd_result1 = query_func1(*bd_args)
        # bd_result2 = query_func2(*bd_args)

    if isinstance(
        bd_result1,
        (
            pd.DataFrame,
            pd.Series,
        ),
    ):
        _test_equal(
            bd_result1,
            pd_result1,
            check_pandas_types=False,
            sort_output=True,
            reset_index=True,
        )
        _test_equal(
            bd_result1,
            pd_result2,
            check_pandas_types=False,
            sort_output=True,
            reset_index=True,
        )
        return

    assert False
    # else:
    #     # For scalar or numeric results
    #     assert isinstance(bd_result1, (int, float)) and isinstance(
    #         bd_result2, (int, float)
    #     )
    #     assert np.isclose(bd_result1, bd_result2)


def test_tpch_q01():
    run_tpch_query_test(tpch.new_tpch_q01, tpch.tpch_q01)


def test_tpch_q02():
    run_tpch_query_test(tpch.new_tpch_q02, tpch.tpch_q02)


def test_tpch_q03():
    run_tpch_query_test(tpch.new_tpch_q03, tpch.tpch_q03)


def test_tpch_q04():
    run_tpch_query_test(tpch.new_tpch_q04, tpch.tpch_q04)


def test_tpch_q05():
    run_tpch_query_test(tpch.new_tpch_q05, tpch.tpch_q05)


def test_tpch_q06():
    run_tpch_query_test(tpch.new_tpch_q06, tpch.tpch_q06, plan_executions=1)


def test_tpch_q07():
    run_tpch_query_test(tpch.new_tpch_q07, tpch.tpch_q07)


def test_tpch_q08():
    run_tpch_query_test(tpch.new_tpch_q08, tpch.tpch_q08)


def test_tpch_q09():
    run_tpch_query_test(tpch.new_tpch_q09, tpch.tpch_q09)


def test_tpch_q10():
    run_tpch_query_test(tpch.new_tpch_q10, tpch.tpch_q10)


def test_tpch_q11():
    run_tpch_query_test(tpch.new_tpch_q11, tpch.tpch_q11, plan_executions=1)


def test_tpch_q12():
    run_tpch_query_test(tpch.new_tpch_q12, tpch.tpch_q12)


def test_tpch_q13():
    run_tpch_query_test(tpch.new_tpch_q13, tpch.tpch_q13)


def test_tpch_q14():
    run_tpch_query_test(tpch.new_tpch_q14, tpch.tpch_q14, plan_executions=2)


def test_tpch_q15():
    run_tpch_query_test(tpch.new_tpch_q15, tpch.tpch_q15, plan_executions=1)


@pytest.mark.skip(
    "RuntimeError: Unsupported expression type in projection 13 (NOT #[207.0])"
)
def test_tpch_q16():
    run_tpch_query_test(tpch.new_tpch_q16, tpch.tpch_q16)


def test_tpch_q17():
    run_tpch_query_test(tpch.new_tpch_q17, tpch.tpch_q17, plan_executions=1)


def test_tpch_q18():
    run_tpch_query_test(tpch.new_tpch_q18, tpch.tpch_q18)


def test_tpch_q19():
    run_tpch_query_test(tpch.new_tpch_q19, tpch.tpch_q19, plan_executions=1)


def test_tpch_q20():
    run_tpch_query_test(tpch.new_tpch_q20, tpch.tpch_q20)


def test_tpch_q21():
    run_tpch_query_test(tpch.new_tpch_q21, tpch.tpch_q21, plan_executions=0)


def test_tpch_q22():
    run_tpch_query_test(tpch.new_tpch_q22, tpch.tpch_q22, plan_executions=1)
