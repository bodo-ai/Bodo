import pandas as pd

import bodo.pandas as bd
from bodo.tests.utils import _test_equal, temp_config_override


def test_from_pandas(datapath):
    """Very simple test to scan a dataframe passed into from_pandas."""

    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": ["a", "b", "c"],
        }
    )
    # Sequential test
    with temp_config_override("dataframe_library_run_parallel", False):
        bdf = bd.from_pandas(df)
        assert bdf._lazy
        assert bdf.plan is not None
        assert bdf.plan.plan_class == "LogicalGetPandasReadSeq"
        duckdb_plan = bdf.plan.generate_duckdb()
        _test_equal(duckdb_plan.df, df)
        _test_equal(
            bdf,
            df,
        )
        assert not bdf._lazy
        assert bdf._mgr._plan is None

    # Parallel test
    bdf = bd.from_pandas(df)
    assert bdf._lazy
    assert bdf.plan is not None
    assert bdf.plan.plan_class == "LogicalGetPandasReadParallel"
    _test_equal(
        bdf,
        df,
    )
    assert not bdf._lazy
    assert bdf._mgr._plan is None
