import pandas as pd
import pytest

import bodo.pandas as bd
from bodo.pandas.plan import assert_executed_plan_count
from bodo.tests.utils import _test_equal


@pytest.fixture
def groupby_df():
    return pd.DataFrame(
        {
            "A": ["A", "B", "A", "B", "A", "B", "A", "B", "A"],
            "C": [1, 0, 1, -2, 1, -2, -1, 2, 1],
            "B": [-2, -2, 3, 4, 5, 6, 4, 8, 9],
        }
    )


def test_basic_agg_udf(groupby_df):
    df = groupby_df
    bdf = bd.from_pandas(df)

    def udf(x):
        if len(x) == 0:
            return None
        return len(x[x > 0]) / len(x)

    df2 = df.groupby(by=["A", "C"]).agg(udf)
    with assert_executed_plan_count(0):
        bdf2 = bdf.groupby(by=["A", "C"]).agg(udf)

    _test_equal(bdf2, df2, check_pandas_types=False, sort_output=True)


def test_basic_agg_multiple_udf(groupby_df):
    df = groupby_df
    bdf = bd.from_pandas(df)

    def udf(x):
        if len(x) == 0:
            return None
        return len(x[x > 0]) / len(x)

    def udf2(x):
        if sum(x) == 0:
            return None
        return sum(x[x < 0]) / sum(x)

    df2 = df.groupby(by=["A"]).agg({"B": udf2, "C": udf})
    with assert_executed_plan_count(0):
        bdf2 = bdf.groupby(by=["A"]).agg({"B": udf2, "C": udf})

    _test_equal(bdf2, df2, check_pandas_types=False, sort_output=True)


def test_agg_mix_udf_builtin(groupby_df):
    df = groupby_df
    bdf = bd.from_pandas(df)

    def udf(x):
        if len(x) == 0:
            return None
        return len(x[x > 0]) / len(x)

    df2 = df.groupby(by=["A"]).agg({"B": sum, "C": udf})
    with assert_executed_plan_count(0):
        bdf2 = bdf.groupby(by=["A"]).agg({"B": sum, "C": udf})

    _test_equal(bdf2, df2, check_pandas_types=False, sort_output=True)
