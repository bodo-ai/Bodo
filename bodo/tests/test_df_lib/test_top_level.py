import pandas as pd
import pytest

import bodo.pandas as bd
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
        pd_obj = nulls_df[col]
        bodo_obj = bdf[col]

        pd_func = getattr(pd_obj, method_name)
        bodo_func = lambda: top_func(bodo_obj)

        pd_error = bodo_error = False
        try:
            out_pd = pd_func()
        except Exception:
            pd_error = True

        try:
            out_bd = bodo_func()
            assert out_bd.is_lazy_plan()
            out_bd = out_bd.execute_plan()
        except Exception:
            bodo_error = True

        assert pd_error == bodo_error
        if pd_error:
            continue

        _test_equal(out_bd, out_pd, check_pandas_types=False, check_names=False)


def test_top_level_to_datetime():
    pdf = pd.DataFrame(
        {
            "dates": ["2021-01-01", "2022-02-15", None, "2023-03-30"],
        }
    )
    bdf = bd.from_pandas(pdf)

    pd_obj = pd.to_datetime(pdf["dates"])
    bd_obj = bd.to_datetime(bdf["dates"])
    assert bd_obj.is_lazy_plan()
    bd_obj = bd_obj.execute_plan()

    _test_equal(bd_obj, pd_obj, check_pandas_types=False, check_names=False)
