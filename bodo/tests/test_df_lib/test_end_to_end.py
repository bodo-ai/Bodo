import operator
import os
import tempfile

import numba
import numpy as np
import pandas as pd
import pytest

import bodo
import bodo.pandas as bd
from bodo.tests.utils import _test_equal, pytest_mark_spawn_mode, temp_config_override

# Various Index kinds to use in test data (assuming maximum size of 100 in input)
MAX_DATA_SIZE = 100


@pytest.fixture(scope="module")
def set_stream_batch_size_three():
    os.environ["BODO_STREAMING_BATCH_SIZE"] = "3"
    yield
    del os.environ["BODO_STREAMING_BATCH_SIZE"]
    # Destroy the spawner and workers so they aren't stuck with this batch size
    bodo.spawn.spawner.destroy_spawner()


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


def test_from_pandas(datapath, index_val, set_stream_batch_size_three):
    """Very simple test to scan a dataframe passed into from_pandas."""

    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 7] * 2,
            "b": [4, 5, 6, 8] * 2,
            "c": ["a", "b", None, "abc"] * 2,
        },
    )
    df.index = index_val[: len(df)]
    # Sequential test
    with temp_config_override("dataframe_library_run_parallel", False):
        bdf = bd.from_pandas(df)
        assert bdf.is_lazy_plan()
        assert bdf._mgr._plan.plan_class == "LogicalGetPandasReadSeq"
        duckdb_plan = bdf._mgr._plan.generate_duckdb()
        _test_equal(duckdb_plan.df, df)
        _test_equal(
            bdf,
            df,
        )
        assert not bdf.is_lazy_plan()
        assert bdf._mgr._plan is None

    # Parallel test
    bdf = bd.from_pandas(df)
    assert bdf.is_lazy_plan()
    assert bdf._mgr._plan.plan_class == "LogicalGetPandasReadParallel"
    _test_equal(
        bdf,
        df,
    )
    assert not bdf.is_lazy_plan()
    assert bdf._mgr._plan is None

    # Make sure projection with a middle column works.
    bdf = bd.from_pandas(df)
    bodo_df2 = bdf["b"]
    df2 = df["b"]
    assert bodo_df2.is_lazy_plan()
    _test_equal(
        bodo_df2,
        df2,
        check_pandas_types=False,
    )


def test_read_parquet(datapath, set_stream_batch_size_three):
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
def test_read_parquet_projection_pushdown(
    datapath, file_path, set_stream_batch_size_three
):
    """Make sure basic projection pushdown works for Parquet read end to end."""
    path = datapath(file_path)

    bodo_out = bd.read_parquet(path)[["three", "four"]]
    py_out = pd.read_parquet(path)[["three", "four"]]

    assert bodo_out.is_lazy_plan()

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
def test_read_parquet_index(df: pd.DataFrame, index_val, set_stream_batch_size_three):
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


def test_read_parquet_len_shape(datapath, set_stream_batch_size_three):
    """Test length/shape after read parquet is correct"""
    path = datapath("example_no_index.parquet")

    bodo_out = bd.read_parquet(path)
    py_out = pd.read_parquet(path)

    assert len(bodo_out) == len(py_out)

    # create a new lazy DF
    bodo_out2 = bd.read_parquet(path)

    # test shape
    assert bodo_out2.shape == py_out.shape


def test_projection(datapath, set_stream_batch_size_three):
    """Very simple test for projection for sanity checking."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1["D"]

    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df2 = py_df1["D"]

    # TODO: remove copy when df.apply(axis=0) is implemented
    # TODO: remove forcing collect when copy() bug with RangeIndex(1) is fixed
    str(bodo_df2)
    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "file_path",
    [
        "dataframe_library/df1.parquet",
        pytest.param("dataframe_library/df1_index.parquet", marks=pytest.mark.skip),
        pytest.param(
            "dataframe_library/df1_multi_index.parquet", marks=pytest.mark.skip
        ),
    ],
)
@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.gt, operator.lt, operator.ge, operator.le]
)
def test_filter_pushdown(datapath, file_path, op, set_stream_batch_size_three):
    """Test for filter with filter pushdown into read parquet."""
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]

    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1[eval(f"bodo_df1.A {op_str} 20")]

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df2.is_lazy_plan()

    pre, post = bd.utils.getPlanStatistics(bodo_df2._mgr._plan)
    _test_equal(pre, 2)
    _test_equal(post, 1)

    py_df1 = pd.read_parquet(datapath(file_path))
    py_df2 = py_df1[eval(f"py_df1.A {op_str} 20")]

    # TODO: remove copy when df.apply(axis=0) is implemented
    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest_mark_spawn_mode
@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.gt, operator.lt, operator.ge, operator.le]
)
def test_filter_distributed(datapath, op):
    bodo.set_verbose_level(2)
    """Very simple test for filter for sanity checking."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))

    @bodo.jit(spawn=True)
    def f(df):
        return df

    # Force plan to execute but keep distributed.
    f(bodo_df1)
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]

    bodo_df2 = bodo_df1[eval(f"bodo_df1.A {op_str} 20")]

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df2.is_lazy_plan()

    py_df2 = py_df1[eval(f"py_df1.A {op_str} 20")]

    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.gt, operator.lt, operator.ge, operator.le]
)
def test_filter(datapath, op, set_stream_batch_size_three):
    """Test for standalone filter."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))

    # Force read parquet node to execute.
    _test_equal(
        bodo_df1.copy(),
        py_df1,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )

    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]

    bodo_df2 = bodo_df1[eval(f"bodo_df1.A {op_str} 20")]

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df2.is_lazy_plan()

    py_df2 = py_df1[eval(f"py_df1.A {op_str} 20")]

    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_filter_multiple1_pushdown(datapath, set_stream_batch_size_three):
    """Test for multiple filter expression."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1[((bodo_df1.A < 20) & (bodo_df1.D > 80))]

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df2.is_lazy_plan()

    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df2 = py_df1[((py_df1.A < 20) & (py_df1.D > 80))]

    # TODO: remove copy when df.apply(axis=0) is implemented
    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.skip(reason="Needs conjunction non-pushdown working.")
def test_filter_multiple1(datapath, set_stream_batch_size_three):
    """Test for multiple filter expression."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))

    # Force read parquet node to execute.
    _test_equal(
        bodo_df1.copy(),
        py_df1,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )

    bodo_df2 = bodo_df1[((bodo_df1.A < 20) & (bodo_df1.D > 80))]
    py_df2 = py_df1[((py_df1.A < 20) & (py_df1.D > 80))]

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df2.is_lazy_plan()

    # TODO: remove copy when df.apply(axis=0) is implemented
    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_filter_string_pushdown(datapath):
    """Test for filtering based on a string pushed down to read parquet."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1[bodo_df1.B == "gamma"]

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df2.is_lazy_plan()

    pre, post = bd.utils.getPlanStatistics(bodo_df2._mgr._plan)
    _test_equal(pre, 2)
    _test_equal(post, 1)

    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df2 = py_df1[py_df1.B == "gamma"]

    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_filter_string(datapath):
    """Test for standalone string filter."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))

    # Force read parquet node to execute.
    _test_equal(
        bodo_df1.copy(),
        py_df1,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )

    bodo_df2 = bodo_df1[bodo_df1.B == "gamma"]

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df2.is_lazy_plan()

    py_df2 = py_df1[py_df1.B == "gamma"]

    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_head_pushdown(datapath):
    """Test for head pushed down to read parquet."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1.head(3)

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df2.is_lazy_plan()

    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df2 = py_df1.head(3)

    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.skip(reason="Non-pushdown physical limit node needs work.")
def test_head(datapath):
    """Test for head pushed down to read parquet."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))

    _test_equal(
        bodo_df1.copy(),
        py_df1,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )

    bodo_df2 = bodo_df1.head(3)
    py_df2 = py_df1.head(3)

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df2.is_lazy_plan()

    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_apply(datapath, index_val, set_stream_batch_size_three):
    """Very simple test for df.apply() for sanity checking."""
    df = pd.DataFrame(
        {
            "a": pd.array([1, 2, 3] * 10, "Int64"),
            "b": pd.array([4, 5, 6] * 10, "Int64"),
            "c": ["a", "b", "c"] * 10,
        },
        index=index_val[:30],
    )
    bdf = bd.from_pandas(df)
    out_pd = df.apply(lambda x: x["a"] + 1, axis=1)
    out_bodo = bdf.apply(lambda x: x["a"] + 1, axis=1)
    _test_equal(out_bodo, out_pd, check_pandas_types=False)


def test_str_lower(datapath, index_val, set_stream_batch_size_three):
    """Very simple test for Series.str.lower for sanity checking."""
    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 7] * 2, "Int64"),
            "B": ["A1", "B1", "C1", "Abc"] * 2,
            "C": pd.array([4, 5, 6, -1] * 2, "Int64"),
        }
    )
    df.index = index_val[: len(df)]
    bdf = bd.from_pandas(df)
    out_pd = df.B.str.lower()
    out_bodo = bdf.B.str.lower()
    assert out_bodo.is_lazy_plan()
    _test_equal(out_bodo, out_pd, check_pandas_types=False)


def test_str_strip(datapath, index_val, set_stream_batch_size_three):
    """Very simple test for Series.str.strip() for sanity checking."""
    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 7], "Int64"),
            "B": ["A1\t", "B1 ", "C1\n", "Abc\t"],
            "C": pd.array([4, 5, 6, -1], "Int64"),
        }
    )
    df.index = index_val[: len(df)]
    bdf = bd.from_pandas(df)
    out_pd = df.B.str.strip()
    out_bodo = bdf.B.str.strip()
    assert out_bodo.is_lazy_plan()
    _test_equal(out_bodo, out_pd, check_pandas_types=False)


def test_chain_python_func(datapath, index_val, set_stream_batch_size_three):
    """Make sure chaining multiple Series functions that run in Python works"""
    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 7], "Int64"),
            "B": ["A1\t", "B1 ", "C1\n", "Abc\t"],
            "C": pd.array([4, 5, 6, -1], "Int64"),
        }
    )
    df.index = index_val[: len(df)]
    bdf = bd.from_pandas(df)
    out_pd = df.B.str.strip().str.lower()
    out_bodo = bdf.B.str.strip().str.lower()
    assert out_bodo.is_lazy_plan()
    _test_equal(out_bodo, out_pd, check_pandas_types=False)


def test_series_map(datapath, index_val, set_stream_batch_size_three):
    """Very simple test for Series.map() for sanity checking."""
    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 7] * 2, "Int64"),
            "B": ["A1", "B1", "C1", "Abc"] * 2,
            "C": pd.array([4, 5, 6, -1] * 2, "Int64"),
        }
    )
    df.index = index_val[: len(df)]

    def func(x):
        return str(x)

    bdf = bd.from_pandas(df)
    out_pd = df.A.map(func)
    out_bodo = bdf.A.map(func)
    assert out_bodo.is_lazy_plan()
    _test_equal(out_bodo, out_pd, check_pandas_types=False)


def test_set_df_column(datapath, index_val, set_stream_batch_size_three):
    """Test setting a dataframe column with a Series function of the same dataframe."""
    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 7], "Int64"),
            "B": ["A1\t", "B1 ", "C1\n", "Abc\t"],
            "C": pd.array([4, 5, 6, -1], "Int64"),
        }
    )
    df.index = index_val[: len(df)]
    bdf = bd.from_pandas(df)

    # Single projection, new column
    bdf["D"] = bdf["B"].str.strip()
    pdf = df.copy()
    pdf["D"] = pdf["B"].str.strip()
    assert bdf.is_lazy_plan()
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Single projection, existing column
    bdf = bd.from_pandas(df)
    bdf["B"] = bdf["B"].str.strip()
    pdf = df.copy()
    pdf["B"] = pdf["B"].str.strip()
    assert bdf.is_lazy_plan()
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Multiple projections, new column
    bdf = bd.from_pandas(df)
    bdf["D"] = bdf["B"].str.strip().map(lambda x: x + "1")
    pdf = df.copy()
    pdf["D"] = pdf["B"].str.strip().map(lambda x: x + "1")
    assert bdf.is_lazy_plan()
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Multiple projections, existing column
    bdf = bd.from_pandas(df)
    bdf["B"] = bdf["B"].str.strip().map(lambda x: x + "1")
    pdf = df.copy()
    pdf["B"] = pdf["B"].str.strip().map(lambda x: x + "1")
    assert bdf.is_lazy_plan()
    _test_equal(bdf, pdf, check_pandas_types=False)


def test_parquet_read_partitioned(datapath, set_stream_batch_size_three):
    """Test reading a partitioned parquet dataset."""
    path = datapath("dataframe_library/example_partitioned.parquet")

    # File generated using:
    # df = pd.DataFrame({
    #                  "a": range(10),
    #                  "b": np.random.randn(10),
    #                  "c": [1, 2] * 5,
    #                  "part": ["a"] * 5 + ["b"] * 5,
    #                  "d": np.arange(10)+1
    #              })
    # df.to_parquet("bodo/tests/data/dataframe_library/example_partitioned.parquet", partition_cols=["part"])

    bodo_out = bd.read_parquet(path)
    py_out = pd.read_parquet(path)

    assert bodo_out.is_lazy_plan()

    # NOTE: Bodo dataframe library currently reads partitioned columns as
    # dictionary-encoded strings but Pandas reads them as categorical.
    _test_equal(
        bodo_out.copy(),
        py_out,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.skip(reason="Parquet partition filter pushdown not yet implemented.")
def test_parquet_read_partitioned_filter(datapath, set_stream_batch_size_three):
    """Test filter pushdown on partitioned parquet dataset."""
    path = datapath("dataframe_library/example_partitioned.parquet")

    bodo_out = bd.read_parquet(path)
    bodo_out = bodo_out[bodo_out.part == "a"]
    py_out = pd.read_parquet(path)
    py_out = py_out[py_out.part == "a"]

    assert bodo_out.is_lazy_plan()
    # TODO: test logs to make sure filter pushdown happened and files skipped

    _test_equal(
        bodo_out,
        py_out,
    )
