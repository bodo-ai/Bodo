import numba  # noqa TID253
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import pytest_mark_one_rank


@pytest_mark_one_rank
@pytest.mark.parametrize(
    "scalar",
    [
        pytest.param(None, id="null"),
        pytest.param(1, id="integer"),
        pytest.param(1.1, id="float"),
        pytest.param("a", id="string"),
    ],
)
def test_coerce_scalar_to_array_unknown(memory_leak_check, scalar):
    """Test that coerce_scalar_to_array works with types.unknown"""

    @bodo.jit
    def f():
        x = bodo.utils.conversion.coerce_scalar_to_array(scalar, 1, numba.types.unknown)
        return pd.Series(x)

    out = f()[0]
    if scalar is None:
        assert out is pd.NA
    else:
        assert out == scalar


@pytest_mark_one_rank
@pytest.mark.parametrize(
    "scalar",
    [
        pytest.param(None, id="null"),
        pytest.param(1, id="integer"),
        pytest.param(1.1, id="float"),
        pytest.param("a", id="string"),
    ],
)
def test_list_to_array_unknown(memory_leak_check, scalar):
    """Test that list_to_array works with types.unknown"""

    @bodo.jit
    def f():
        x = bodo.utils.conversion.list_to_array([scalar], numba.types.unknown)
        return pd.Series(x)

    out = f()[0]
    if scalar is None:
        assert out is pd.NA
    else:
        assert out == scalar
