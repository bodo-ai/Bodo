import pandas as pd
import numpy as np
import numba
import bodo
from bodo.utils.typing import BodoError
import pytest


def test_isin():
    """
    tests error for 'values' argument of Series.isin()
    """

    def impl(S, values):
        return S.isin(values)

    S = pd.Series([3, 2, np.nan, 2, 7], [3, 4, 2, 1, 0], name="A")
    with pytest.raises(BodoError, match="parameter should be a set or a list"):
        bodo.jit(impl)(S, "3")
