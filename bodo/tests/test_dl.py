import numpy as np
import pytest

import bodo
from bodo.utils.typing import BodoError


def test_error_checking():
    """ Test that bodo.prepare_data() throws error with replicated data """

    def impl(x, y):
        x, y = bodo.dl.prepare_data(x, y)
        return x, y

    X = np.arange(10)
    y = np.arange(10)
    with pytest.raises(
        BodoError, match="Arguments of bodo.dl.prepare_data are not distributed"
    ):
        bodo.jit(impl)(X, y)
