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


@pytest.fixture(
    params=(
        pytest.param(bodo.jit, id="jit_no_kwargs"),
        pytest.param(bodo.jit(spawn=False, distributed=False), id="jit_no_spawn"),
        pytest.param(bodo.jit(cache=True), id="jit_with_cache"),
    ),
    scope="module",
)
def engine(request):
    return request.param


def test_apply_basic(engine):
    """Simplest test to check Pandas UDF apply hook is set up properly"""

    df = pd.DataFrame({"A": np.arange(30)})

    bodo_result = df.apply(lambda x: x.A, axis=1, engine=engine)

    pandas_result = df.apply(lambda x: x.A, axis=1)

    _test_equal(bodo_result, pandas_result, check_pandas_types=False)


def test_apply_raw_error():
    """Test passing raw=True raises appropriate error message."""

    df = pd.DataFrame({"A": np.arange(30)})

    with pytest.raises(
        ValueError,
        match="BodoExecutionEngine: does not support the raw=True for DataFrame.apply.",
    ):
        df.apply(lambda x: x.A, axis=1, engine=bodo.jit, raw=True)


def test_udf_args(engine):
    df = pd.DataFrame({"A": np.arange(30)})

    def udf(x, a):
        return x.A + a

    bodo_result = df.apply(udf, axis=1, engine=engine, args=(1,))

    pandas_result = df.apply(udf, axis=1, args=(1,))

    _test_equal(bodo_result, pandas_result, check_pandas_types=False)


def test_udf_kwargs(engine):
    df = pd.DataFrame({"A": np.arange(30), "B": ["hi", "hello", "goodbye"] * 10})

    def udf(x, a=1, b="goodbye", d=3):
        if b == x.B:
            return x.A + a
        else:
            return x.A + d

    bodo_out = df.apply(udf, axis=1, args=(4,), d=16, b="hi", engine=engine)

    pandas_out = df.apply(udf, axis=1, args=(4,), d=16, b="hi")

    _test_equal(bodo_out, pandas_out, check_pandas_types=False)


def test_udf_cache():
    """Tests that we can call the same UDF multiple times with cache flag on
    without any errors. TODO: check cache."""
    engine = bodo.jit(cache=True)

    df = pd.DataFrame({"A": np.arange(30)})

    def udf(x, a):
        return x.A + a

    bodo_result = df.apply(udf, axis=1, engine=engine, args=(1,))

    pandas_result = df.apply(udf, axis=1, args=(1,))

    _test_equal(bodo_result, pandas_result, check_pandas_types=False)

    bodo_result = df.apply(udf, axis=1, engine=engine, args=(1,))

    _test_equal(bodo_result, pandas_result, check_pandas_types=False)


def test_udf_str(engine):
    """Test passing string as func works properly."""
    df = pd.DataFrame({"A": np.arange(30)})

    str_func = "mean"

    bodo_out = df.apply(str_func, axis=1, engine=engine)

    pandas_out = df.apply(str_func, axis=1)

    _test_equal(bodo_out, pandas_out, check_pandas_types=False)


def test_udf_map(engine):
    """Test basic map support with bodo engine."""
    ser = pd.Series(np.arange(30))

    def udf(x):
        return str(x + 10)

    bodo_out = ser.map(udf, engine=engine)

    pandas_out = ser.map(udf)

    _test_equal(bodo_out, pandas_out, check_pandas_types=False)


def test_udf_map_unsupported():
    """Test passing unsupported arguments to map raises appropriate errors."""
    ser = pd.Series([1, 2, 3, 4, None] * 2)
    engine = bodo.jit

    def udf(x, y=2):
        if x is None:
            return None
        return str(x + y)

    # additional kwargs are not supported
    with pytest.raises(ValueError, match=r"BodoExecutionEngine:.*"):
        ser.map(udf, y=1, engine=engine)

    # na_action='ignore' not supported
    with pytest.raises(ValueError, match=r"BodoExecutionEngine:.*"):
        ser.map(udf, na_action="ignore", engine=engine)
