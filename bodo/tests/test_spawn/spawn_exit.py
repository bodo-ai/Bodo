"""Simple spawn program to test atexit behavior works as expected.
See test_spawn_mode.py::test_spawn_atexit_delete_result
"""

import pandas as pd

import bodo


@bodo.jit(spawn=True)
def _get_bodo_dataframe(df):
    return df


df = pd.DataFrame({"A": [1, 2, 3, 4, 5] * 100})

bodo_df = _get_bodo_dataframe(df)
