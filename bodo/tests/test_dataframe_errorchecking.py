import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.utils.typing import BodoError


@pytest.mark.slow
def test_df_head_errors(memory_leak_check):
    def impl():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.head(5.0)

    with pytest.raises(BodoError, match="Dataframe.head\(\): 'n' must be an Integer"):
        bodo.jit(impl)()


@pytest.mark.slow
def test_df_tail_errors(memory_leak_check):
    def impl():
        df = pd.DataFrame({"A": np.random.randn(10), "B": np.arange(10)})
        return df.tail(5.0)

    with pytest.raises(BodoError, match="Dataframe.tail\(\): 'n' must be an Integer"):
        bodo.jit(impl)()
