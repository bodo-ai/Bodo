import pandas as pd
import pytest

import bodo
import bodo.pandas as bd
from bodo.tests.conftest import datapath_util
from bodo.tests.utils import _test_equal, pytest_mark_spawn_mode

pytestmark = pytest.mark.jit_dependency


def df_from_pandas():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": ["a", "b", "c"],
        }
    )
    bdf = bd.from_pandas(df)
    return df, bdf


def df_from_parquet():
    path = datapath_util("example_no_index.parquet")

    return pd.read_parquet(path, dtype_backend="pyarrow"), bd.read_parquet(path)


@pytest.fixture(params=[df_from_pandas, df_from_parquet])
def df_gen(request):
    df, bdf = request.param()

    yield df, bdf


def test_bodo_df_to_jit(df_gen):
    df, bdf = df_gen

    @bodo.jit(spawn=False)
    def f(df):
        return df

    _test_equal(f(bdf), df)


@pytest_mark_spawn_mode
def test_bodo_df_to_jit_spawn(df_gen):
    df, bdf = df_gen

    @bodo.jit(spawn=True)
    def f(df):
        return df

    _test_equal(f(bdf), df)
