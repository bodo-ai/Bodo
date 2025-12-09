import pandas as pd
import pytest

import bodo.pandas as bd
from bodo.pandas.plan import assert_executed_plan_count
from bodo.tests.utils import _test_equal


@pytest.fixture
def nulls_df():
    return pd.DataFrame(
        {
            "A": [1, None, 3],
            "B": ["x", None, "z"],
            "C": [None, None, None],
        }
    )


@pytest.mark.parametrize(
    "top_func, method_name",
    [
        (bd.isna, "isna"),
        (bd.isnull, "isnull"),
        (bd.notna, "notna"),
        (bd.notnull, "notnull"),
    ],
)
def test_top_level_redirects(nulls_df, top_func, method_name):
    bdf = bd.from_pandas(nulls_df)
    for col in bdf.columns:
        with assert_executed_plan_count(0):
            pd_obj = nulls_df[col]
            bodo_obj = bdf[col]

            pd_func = getattr(pd_obj, method_name)

            def bodo_func():
                return top_func(bodo_obj)

            out_pd = pd_func()
            out_bd = bodo_func()
        with assert_executed_plan_count(1):
            _ = out_bd.execute_plan()
        _test_equal(out_bd, out_pd, check_pandas_types=False, check_names=False)


def test_top_level_to_datetime():
    with assert_executed_plan_count(0):
        # Single column string case
        pdf1 = pd.DataFrame({"dates": ["2021-01-01", "2022-02-15", None, "2023-03-30"]})
        bdf1 = bd.from_pandas(pdf1)
        pd_obj1 = pd.to_datetime(pdf1["dates"], utc=True)
        bd_obj1 = bd.to_datetime(bdf1["dates"], utc=True)
    with assert_executed_plan_count(1):
        _ = bd_obj1.execute_plan()
    _test_equal(bd_obj1, pd_obj1, check_pandas_types=False, check_names=False)

    with assert_executed_plan_count(0):
        # Multi-column case: year, month, day
        pdf2 = pd.DataFrame(
            {
                "year": [2015, 2016, 2017],
                "month": [2, 3, 4],
                "day": [4, 5, 6],
            }
        )
        bdf2 = bd.from_pandas(pdf2)
        pd_obj2 = pd.to_datetime(pdf2)
        bd_obj2 = bd.to_datetime(bdf2)
    with assert_executed_plan_count(1):
        _ = bd_obj2.execute_plan()
    _test_equal(bd_obj2, pd_obj2, check_pandas_types=False, check_names=False)

    with assert_executed_plan_count(0):
        # With NaNs
        pdf3 = pd.DataFrame(
            {
                "year": [2020, None, 2022],
                "month": [1, 2, None],
                "day": [10, 20, 30],
            }
        )
        bdf3 = bd.from_pandas(pdf3)
        pd_obj3 = pd.to_datetime(pdf3)
        bd_obj3 = bd.to_datetime(bdf3)
    _test_equal(
        bd_obj3.execute_plan(), pd_obj3, check_pandas_types=False, check_names=False
    )

    # Format not specified
    with assert_executed_plan_count(1):
        pdf1 = pd.DataFrame({"dates": ["2021-01-01", "2022-02-15", None, "2023-03-30"]})
        bdf1 = bd.from_pandas(pdf1)
        pd_obj1 = pd.to_datetime(pdf1["dates"])
        bd_obj1 = bd.to_datetime(bdf1["dates"])
    with assert_executed_plan_count(1):
        _ = bd_obj1.execute_plan()
    _test_equal(bd_obj1, pd_obj1, check_pandas_types=False, check_names=False)

    # Format has timezone info
    with assert_executed_plan_count(1):
        pdf1 = pd.DataFrame(
            {
                "dates": [
                    "2021-01-01+01:00",
                    "2022-02-15+01:00",
                    None,
                    "2023-03-30+01:00",
                ]
            }
        )
        bdf1 = bd.from_pandas(pdf1)
        pd_obj1 = pd.to_datetime(pdf1["dates"], format="%Y-%m-%d%z")
        bd_obj1 = bd.to_datetime(bdf1["dates"], format="%Y-%m-%d%z")
    with assert_executed_plan_count(1):
        _ = bd_obj1.execute_plan()
    _test_equal(bd_obj1, pd_obj1, check_pandas_types=False, check_names=False)

    # Format doesn't have timezone info
    with assert_executed_plan_count(0):
        pdf1 = pd.DataFrame({"dates": ["2021-01-01", "2022-02-15", None, "2023-03-30"]})
        bdf1 = bd.from_pandas(pdf1)
        pd_obj1 = pd.to_datetime(pdf1["dates"], format="%Y-%m-%d")
        bd_obj1 = bd.to_datetime(bdf1["dates"], format="%Y-%m-%d")
    with assert_executed_plan_count(1):
        _ = bd_obj1.execute_plan()
    _test_equal(bd_obj1, pd_obj1, check_pandas_types=False, check_names=False)
