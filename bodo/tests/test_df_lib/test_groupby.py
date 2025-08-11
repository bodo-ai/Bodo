import pandas as pd

import bodo.pandas as bd
from bodo.pandas.plan import assert_executed_plan_count
from bodo.tests.utils import _test_equal


def test_basic_agg_udf():
    df = pd.DataFrame(
        {"A": [1, 2, 1, 2, 1, 2, 1, 2, 1], "B": [-2, -2, 3, 4, 5, 6, 4, 8, 9]}
    )
    bdf = bd.from_pandas(df)

    df2 = df.groupby(by="A").agg(lambda x: len(x[x > 0]) / len(x))
    with assert_executed_plan_count(0):
        bdf2 = bdf.groupby(by="A").agg(lambda x: len(x[x > 0]) / len(x))

    _test_equal(bdf2, df2, check_pandas_types=False)
