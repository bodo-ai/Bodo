import pandas as pd

import bodo
import bodo.pandas as bd
from bodo.tests.utils import _test_equal, pytest_mark_spawn_mode


def test_bodo_df_to_jit():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": ["a", "b", "c"],
        }
    )
    bdf = bd.from_pandas(df)

    @bodo.jit(spawn=False)
    def f(df):
        return df

    _test_equal(f(bdf), df)


@pytest_mark_spawn_mode
def test_bodo_df_to_jit_spawn():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": ["a", "b", "c"],
        }
    )
    bdf = bd.from_pandas(df)

    @bodo.jit(spawn=True)
    def f(df):
        return df

    _test_equal(f(bdf), df)
