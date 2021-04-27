import pandas as pd
import pytest

import bodo
from bodo.utils.typing import BodoError


def test_undefined_variable():
    message = "name 'undefined_variable' is not defined"
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: pd.read_csv(undefined_variable))()
