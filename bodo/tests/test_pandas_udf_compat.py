"""
This file tests the Bodo implementation of the Pandas UDF interface.
See https://github.com/pandas-dev/pandas/pull/61032 for more details.

This feature is only availible on newer versions of Pandas (>=3.0)
"""

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.pandas_compat import pandas_version
from bodo.tests.utils import _test_equal, pytest_spawn_mode

pytestmark = [
    pytest.mark.skipif(
        pandas_version < (3, 0), reason="Third-party UDF engines requires Pandas >= 3.0"
    )
] + pytest_spawn_mode


def test_basic_apply_example():
    """Simplest test to check Pandas UDF apply hook is set up properly"""

    df = pd.DataFrame({"A": np.arange(30)})

    bodo_result = df.apply(lambda x: x.A, axis=1, engine=bodo.jit)

    pandas_result = df.apply(lambda x: x.A, axis=1)

    _test_equal(bodo_result, pandas_result, check_pandas_types=False)
