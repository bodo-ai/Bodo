import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import bodo.pandas as bd
from bodo.tests.utils import _test_equal, temp_config_override

# Various Index kinds to use in test data (assuming maximum size of 100 in input)
MAX_DATA_SIZE = 100


@pytest.fixture(
    params=[
        pd.RangeIndex(MAX_DATA_SIZE),
        pd.date_range("1998-01-01", periods=MAX_DATA_SIZE),
        pd.MultiIndex.from_arrays(
            (np.arange(MAX_DATA_SIZE) * 2, np.arange(MAX_DATA_SIZE) * 4),
            names=["first", "second"],
        ),
    ]
)
def index_val(request):
    return request.param


def test_from_pandas(datapath, index_val):
    """Very simple test to scan a dataframe passed into from_pandas."""

    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": ["a", "b", "c"],
        },
        index=index_val[:3],
    )
    # Sequential test
    with temp_config_override("dataframe_library_run_parallel", False):
        bdf = bd.from_pandas(df)
        assert bdf._lazy
        assert bdf.plan is not None
        assert bdf.plan.plan_class == "LogicalGetPandasReadSeq"
        duckdb_plan = bdf.plan.generate_duckdb()
        _test_equal(duckdb_plan.df, df)
        _test_equal(
            bdf,
            df,
        )
        assert not bdf._lazy
        assert bdf._mgr._plan is None

    # Parallel test
    bdf = bd.from_pandas(df)
    assert bdf._lazy
    assert bdf.plan is not None
    assert bdf.plan.plan_class == "LogicalGetPandasReadParallel"
    _test_equal(
        bdf,
        df,
    )
    assert not bdf._lazy
    assert bdf._mgr._plan is None


def test_read_parquet(datapath):
    """Very simple test to read a parquet file for sanity checking."""
    path = datapath("example_no_index.parquet")

    bodo_out = bd.read_parquet(path)
    py_out = pd.read_parquet(path)

    _test_equal(
        bodo_out,
        py_out,
    )


@pytest.mark.parametrize(
    "file_path",
    [
        "example_no_index.parquet",
        "example_single_index.parquet",
        "example_multi_index.parquet",
    ],
)
def test_read_parquet_projection_pushdown(datapath, file_path):
    """Make sure basic projection pushdown works for Parquet read end to end."""
    path = datapath(file_path)

    bodo_out = bd.read_parquet(path)[["three", "four"]]
    py_out = pd.read_parquet(path)[["three", "four"]]

    _test_equal(
        bodo_out,
        py_out,
    )


@pytest.mark.parametrize(
    "df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "one": [-1.0, np.nan, 2.5, 3.0, 4.0, 6.0, 10.0],
                    "two": ["foo", "bar", "baz", "foo", "bar", "baz", "foo"],
                    "three": [True, False, True, True, True, False, False],
                    "four": [-1.0, 5.1, 2.5, 3.0, 4.0, 6.0, 11.0],
                    "five": ["foo", "bar", "baz", None, "bar", "baz", "foo"],
                }
            ),
            id="df1",
        )
    ],
)
def test_read_parquet_index(df: pd.DataFrame, index_val):
    """Test reading parquet with index column works as expected."""
    df.index = index_val[: len(df)]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "example.pq")

        df.to_parquet(path)

        bodo_out = bd.read_parquet(path)
        py_out = pd.read_parquet(path)

        _test_equal(
            bodo_out,
            py_out,
        )


def test_read_parquet_len_shape(datapath):
    """Test length/shape after read parquet is correct"""
    path = datapath("example_no_index.parquet")

    bodo_out = bd.read_parquet(path)
    py_out = pd.read_parquet(path)

    assert len(bodo_out) == len(py_out)

    # create a new lazy DF
    bodo_out2 = bd.read_parquet(path)

    # test shape
    assert bodo_out2.shape == py_out.shape


def test_projection(datapath):
    """Very simple test for projection for sanity checking."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1["D"]

    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df2 = py_df1["D"]

    _test_equal(bodo_df2, py_df2, check_pandas_types=False)


def test_filter(datapath):
    """Very simple test for filter for sanity checking."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1[bodo_df1.A < 20]

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df2._lazy
    assert bodo_df2.plan is not None

    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df2 = py_df1[py_df1.A < 20]

    _test_equal(bodo_df2, py_df2, check_pandas_types=False)


def test_apply(datapath):
    """Very simple test for df.apply() for sanity checking."""
    df = pd.DataFrame(
        {
            "a": pd.array([1, 2, 3] * 10, "Int64"),
            "b": pd.array([4, 5, 6] * 10, "Int64"),
            "c": ["a", "b", "c"] * 10,
        }
    )
    bdf = bd.from_pandas(df)
    out_pd = df.apply(lambda x: x["a"] + 1, axis=1)
    out_bodo = bdf.apply(lambda x: x["a"] + 1, axis=1)
    _test_equal(out_bodo, out_pd, check_pandas_types=False)


def test_str_lower(datapath):
    """Very simple test for Series.str.lower for sanity checking."""
    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3], "Int64"),
            "B": ["A1", "B1", "C1"],
            "C": pd.array([4, 5, 6], "Int64"),
        }
    )
    bdf = bd.from_pandas(df)
    out_pd = df.B.str.lower()
    out_bodo = bdf.B.str.lower()
    assert out_bodo._lazy
    assert out_bodo.plan is not None
    _test_equal(out_bodo, out_pd, check_pandas_types=False)
