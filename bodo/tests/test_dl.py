import numpy as np
import pytest

import bodo
from bodo.utils.typing import BodoError


@pytest.mark.slow
def test_error_checking():
    """ Test that bodo.prepare_data() throws error with replicated data """

    def impl(x):
        x = bodo.dl.prepare_data(x)
        return x

    X = np.arange(10)
    with pytest.raises(
        BodoError, match="Argument of bodo.dl.prepare_data is not distributed"
    ):
        bodo.jit(impl)(X)
