import pandas as pd
import numpy as np
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


def test_series_dt_not_supported():
    """
    tests error for unsupported Series.dt methods
    """

    def impl():
        rng = pd.date_range("20/02/2020", periods=5, freq="M")
        ts = pd.Series(np.random.randn(len(rng)), index=rng)
        ps = ts.to_period()
        return ps

    with pytest.raises(BodoError, match="not supported"):
        bodo.jit(impl)()
