import pandas as pd

import bodo.pandas as bd
from bodo.tests.utils import _test_equal


def test_from_pandas(datapath):
    """Very simple test to scan a dataframe passed into from_pandas."""

    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": ["a", "b", "c"],
        }
    )
    bdf = bd.from_pandas(df)
    assert bdf._lazy
    assert bdf.plan is not None
    duckdb_plan = bdf.plan.generate_duckdb()
    _test_equal(duckdb_plan.df, df)

    _test_equal(
        bdf,
        df,
    )
