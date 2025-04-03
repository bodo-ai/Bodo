import pandas as pd

import bodo.pandas as bd
from bodo.tests.utils import _test_equal


def test_from_pandas(datapath):
    """Very simple test to read a parquet file for sanity checking."""

    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": ["a", "b", "c"],
        }
    )
    bdf = bd.from_pandas(df)

    _test_equal(
        bdf,
        df,
    )
