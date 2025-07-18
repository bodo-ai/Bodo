import numpy as np
import pandas as pd
import pytest

import benchmarks.tpch.dataframe_lib as tpch
import bodo.pandas as bd
from bodo.tests.utils import _test_equal

datapath = "bodo/tests/data/tpch-test_data/parquet"


def run_tpch_query_test(query_func):
    pd_args = [
        getattr(tpch, f"load_{key}")(datapath)
        for key in tpch._query_to_datasets[int(query_func.__name__[-2:])]
    ]
    bd_args = [bd.from_pandas(df) for df in pd_args]

    pd_kwargs = {"pd": pd}
    pd_result = query_func(*pd_args, **pd_kwargs)
    bd_result = query_func(*bd_args)

    if isinstance(
        pd_result,
        (
            pd.DataFrame,
            pd.Series,
        ),
    ):
        _test_equal(bd_result, pd_result, check_pandas_types=False, reset_index=True)
    else:
        # For scalar or numeric results
        assert isinstance(pd_result, (int, float)) and isinstance(
            bd_result, (int, float)
        )
        assert np.isclose(pd_result, bd_result)


def test_tpch_q01():
    run_tpch_query_test(tpch.tpch_q01)


def test_tpch_q02():
    run_tpch_query_test(tpch.tpch_q02)


def test_tpch_q03():
    run_tpch_query_test(tpch.tpch_q03)


@pytest.mark.skip(reason="TypeError: a bytes-like object is required, not 'str'")
def test_tpch_q04():
    run_tpch_query_test(tpch.tpch_q04)


def test_tpch_q05():
    run_tpch_query_test(tpch.tpch_q05)


def test_tpch_q06():
    run_tpch_query_test(tpch.tpch_q06)


def test_tpch_q07():
    run_tpch_query_test(tpch.tpch_q07)


def test_tpch_q08():
    run_tpch_query_test(tpch.tpch_q08)


def test_tpch_q09():
    run_tpch_query_test(tpch.tpch_q09)


def test_tpch_q10():
    run_tpch_query_test(tpch.tpch_q10)


def test_tpch_q11():
    run_tpch_query_test(tpch.tpch_q11)


def test_tpch_q12():
    run_tpch_query_test(tpch.tpch_q12)


def test_tpch_q13():
    run_tpch_query_test(tpch.tpch_q13)


def test_tpch_q14():
    run_tpch_query_test(tpch.tpch_q14)


@pytest.mark.skip(reason="Length mismatch")
def test_tpch_q15():
    run_tpch_query_test(tpch.tpch_q15)


@pytest.mark.skip(reason="Waiting on dropduplicates to produce BodoDataFrame")
def test_tpch_q16():
    run_tpch_query_test(tpch.tpch_q16)


def test_tpch_q17():
    run_tpch_query_test(tpch.tpch_q17)


def test_tpch_q18():
    run_tpch_query_test(tpch.tpch_q18)


def test_tpch_q19():
    run_tpch_query_test(tpch.tpch_q19)


def test_tpch_q20():
    run_tpch_query_test(tpch.tpch_q20)


def test_tpch_q21():
    run_tpch_query_test(tpch.tpch_q21)


def test_tpch_q22():
    run_tpch_query_test(tpch.tpch_q22)
