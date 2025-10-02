import pandas as pd
import pytest

import bodo.pandas as bd
from bodo.pandas.plan import assert_executed_plan_count
from bodo.tests.utils import _test_equal

pytestmark = pytest.mark.jit_dependency


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


@pytest.mark.jit_dependency
def test_top_level_to_datetime():
    with assert_executed_plan_count(0):
        # Single column string case
        pdf1 = pd.DataFrame({"dates": ["2021-01-01", "2022-02-15", None, "2023-03-30"]})
        bdf1 = bd.from_pandas(pdf1)
        pd_obj1 = pd.to_datetime(pdf1["dates"])
        bd_obj1 = bd.to_datetime(bdf1["dates"])
    with assert_executed_plan_count(1):
        _ = bd_obj1.execute_plan()

    # Remove after testing!
    import bodo

    @bodo.jit
    def t1(x):
        return x.map(lambda v: pd.to_datetime(v), na_action="ignore")

    t1res = t1(pdf1["dates"])
    print("t1res===================\n", t1res, t1res.isna())
    pre_datetime_null = bd.from_pandas(pdf1)
    print(
        "pre_datetime_null===================\n",
        pre_datetime_null,
        pre_datetime_null.isna(),
    )
    t2 = bd.Series(pdf1["dates"])
    # t3 = t2.map(lambda x: 7.7, na_action='ignore')
    t3 = t2.map(lambda x: pd.Timestamp(year=2024, month=1, day=1), na_action="ignore")
    print("t3", t3, t3.isna())
    # t2 = bd.Series(pd_obj1)
    # t3 = t2.map(lambda x: x, na_action='ignore')
    # print("t3", t3, t3.isna())
    print("pd_obj1\n", pd_obj1, pd_obj1.isna())
    print("bd_obj1\n", bd_obj1, bd_obj1.isna())
    # pd_obj1_with_bd_mask = pd_obj1.mask(bd_obj1.isna())

    _test_equal(bd_obj1, pd_obj1, check_pandas_types=False, check_names=False)

    """
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
    """
