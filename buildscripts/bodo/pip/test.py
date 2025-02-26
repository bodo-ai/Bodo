"""
Part of CIBW test command that runs on all pip wheels.
"""

import pandas as pd

import bodo

pd.DataFrame({"a": [1, 2, 3]}).to_parquet("test.parquet")


@bodo.jit
def f():
    df = pd.read_parquet("test.parquet")
    return df.a.sum()


assert f() == 6
