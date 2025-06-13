import operator
import os
import tempfile

import numba
import numpy as np
import pandas as pd
import pytest

import bodo
import bodo.pandas as bd
from bodo.pandas.utils import BodoLibFallbackWarning
from bodo.tests.utils import _test_equal, pytest_mark_spawn_mode, temp_config_override

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
    # len directly on parquet file doesn't require plan execution
    assert bodo_out.is_lazy_plan()

    # create a new lazy DF
    bodo_out2 = bd.read_parquet(path)

    # test shape
    assert bodo_out2.shape == py_out.shape
    # shape directly on parquet file doesn't require plan execution
    assert bodo_out2.is_lazy_plan()


def test_read_parquet_series_len_shape(datapath):
    """Test length/shape after read parquet is correct"""
    path = datapath("dataframe_library/df1.parquet")

    bodo_out = bd.read_parquet(path)
    bodo_out = bodo_out["A"]
    py_out = pd.read_parquet(path)
    py_out = py_out["A"]

    assert len(bodo_out) == len(py_out)
    # len directly on parquet file doesn't require plan execution
    assert bodo_out.is_lazy_plan()

    # test shape
    assert bodo_out.shape == py_out.shape
    # shape directly on parquet file doesn't require plan execution
    assert bodo_out.is_lazy_plan()


def test_write_parquet(index_val):
    """Test writing a DataFrame to parquet."""
    df = pd.DataFrame(
        {
            "one": [-1.0, np.nan, 2.5, 3.0, 4.0, 6.0, 10.0],
            "two": ["foo", "bar", "baz", "foo", "bar", "baz", "foo"],
            "three": [True, False, True, True, True, False, False],
            "four": [-1.0, 5.1, 2.5, 3.0, 4.0, 6.0, 11.0],
            "five": ["foo", "bar", "baz", None, "bar", "baz", "foo"],
        }
    )
    df.index = index_val[: len(df)]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_write.parquet")

        bodo_df = bd.from_pandas(df)
        bodo_df.to_parquet(path)
        assert bodo_df.is_lazy_plan()

        # Read back to check
        py_out = pd.read_parquet(path)
        _test_equal(
            py_out,
            df,
            check_pandas_types=False,
            sort_output=True,
        )


def test_projection(datapath):
    """Very simple test for projection for sanity checking."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1["D"]

    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df2 = py_df1["D"]

    # TODO: remove copy when df.apply(axis=0) is implemented
    # TODO: remove forcing collect when copy() bug with RangeIndex(1) is fixed
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
        "dataframe_library/df1_index.parquet",
        "dataframe_library/df1_multi_index.parquet",
    ],
)
@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.gt, operator.lt, operator.ge, operator.le]
)
def test_filter_pushdown(datapath, file_path, op):
    """Test for filter with filter pushdown into read parquet."""
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]

    bodo_df1 = bd.read_parquet(datapath(file_path))
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
    "file_path",
    [
        "dataframe_library/df1.parquet",
        "dataframe_library/df1_index.parquet",
        "dataframe_library/df1_multi_index.parquet",
    ],
)
@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.gt, operator.lt, operator.ge, operator.le]
)
def test_filter_distributed(datapath, file_path, op):
    """Very simple test for filter for sanity checking."""
    bodo_df1 = bd.read_parquet(datapath(file_path))
    py_df1 = pd.read_parquet(datapath(file_path))

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
        reset_index=False,
    )


@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.gt, operator.lt, operator.ge, operator.le]
)
def test_filter(datapath, op):
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
        reset_index=False,
    )


def test_filter_multiple1_pushdown(datapath):
    """Test for multiple filter expression."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1[((bodo_df1.A < 20) & ~(bodo_df1.D > 80))]

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df2.is_lazy_plan()

    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df2 = py_df1[((py_df1.A < 20) & ~(py_df1.D > 80))]

    # TODO: remove copy when df.apply(axis=0) is implemented
    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_filter_multiple1(datapath):
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

    bodo_df2 = bodo_df1[((bodo_df1.A < 20) & ~(bodo_df1.D > 80))]
    py_df2 = py_df1[((py_df1.A < 20) & ~(py_df1.D > 80))]

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


@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.gt, operator.lt, operator.ge, operator.le]
)
def test_filter_datetime_pushdown(datapath, op):
    """Test for standalone filter."""
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1[
        eval(f"bodo_df1.F {op_str} pd.to_datetime('2025-07-17 22:39:02')")
    ]

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df2.is_lazy_plan()

    pre, post = bd.utils.getPlanStatistics(bodo_df2._mgr._plan)
    _test_equal(pre, 2)
    _test_equal(post, 1)

    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df2 = py_df1[eval(f"py_df1.F {op_str} pd.to_datetime('2025-07-17 22:39:02')")]

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
def test_filter_datetime(datapath, op):
    """Test for standalone filter."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))

    # Force read parquet node to execute so the filter doesn't get pushed into the read.
    _test_equal(
        bodo_df1.copy(),
        py_df1,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )

    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]

    bodo_df2 = bodo_df1[
        eval(f"bodo_df1.F {op_str} pd.to_datetime('2025-07-17 22:39:02')")
    ]

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df2.is_lazy_plan()

    py_df2 = py_df1[eval(f"py_df1.F {op_str} pd.to_datetime('2025-07-17 22:39:02')")]

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

    pre, post = bd.utils.getPlanStatistics(bodo_df2._plan)
    _test_equal(pre, 2)
    _test_equal(post, 1)

    # Contents not guaranteed to be the same as Pandas so just check length.
    assert len(bodo_df2) == 3


def test_projection_head_pushdown(datapath):
    """Test for projection and head pushed down to read parquet."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1["D"]
    bodo_df3 = bodo_df2.head(3)

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df3.is_lazy_plan()

    # Contents not guaranteed to be the same as Pandas so just check length.
    assert len(bodo_df3) == 3


def test_series_head(datapath):
    """Test for Series.head() reading from Pandas."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1["D"]
    bodo_df2.execute_plan()
    bodo_df3 = bodo_df2.head(3)

    # Make sure bodo_df3 is unevaluated at this point.
    assert bodo_df3.is_lazy_plan()

    # Contents not guaranteed to be the same as Pandas so just check length.
    assert len(bodo_df3) == 3


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

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df2.is_lazy_plan()

    # Contents not guaranteed to be the same as Pandas so just check length.
    assert len(bodo_df2) == 3


def test_apply(datapath, index_val):
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


def test_chain_python_func(datapath, index_val):
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


@pytest.mark.parametrize(
    "na_action",
    [
        pytest.param(None, id="na_action_none"),
        pytest.param("ignore", id="na_action_ignore"),
    ],
)
def test_series_map(datapath, index_val, na_action):
    """Very simple test for Series.map() for sanity checking."""
    df = pd.DataFrame(
        {
            "A": pd.array([None, None, 3, 7, 2] * 2, "Int64"),
            "B": [None, None, "B1", "C1", "Abc"] * 2,
            "C": pd.array([4, 5, 6, -1, 1] * 2, "Int64"),
        }
    )
    df.index = index_val[: len(df)]

    def func(x):
        return str(x)

    bdf = bd.from_pandas(df)
    out_pd = df.A.map(func, na_action=na_action)
    out_bodo = bdf.A.map(func, na_action=na_action)
    assert out_bodo.is_lazy_plan()
    _test_equal(out_bodo, out_pd, check_pandas_types=False)


def test_set_df_column(datapath, index_val):
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

    # Trivial case: set a column to existing column
    bdf = bd.from_pandas(df)
    bdf["D"] = bdf["B"]
    pdf = df.copy()
    pdf["D"] = pdf["B"]
    assert bdf.is_lazy_plan()
    _test_equal(bdf, pdf, check_pandas_types=False)


def test_set_df_column_const(datapath, index_val):
    """Test setting a dataframe column with a constant value."""
    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 7], "Int64"),
            "B": ["A1\t", "B1 ", "C1\n", "Abc\t"],
            "C": pd.array([4, 5, 6, -1], "Int64"),
        }
    )
    df.index = index_val[: len(df)]
    bdf = bd.from_pandas(df)

    # New integer column
    bdf["D"] = 111
    pdf = df.copy()
    pdf["D"] = 111
    assert bdf.is_lazy_plan()
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Replace existing column with float
    bdf = bd.from_pandas(df)
    bdf["B"] = 1.23
    pdf = df.copy()
    pdf["B"] = 1.23
    assert bdf.is_lazy_plan()
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Replace existing column with string
    bdf = bd.from_pandas(df)
    bdf["C"] = "ABC"
    pdf = df.copy()
    pdf["C"] = "ABC"
    assert bdf.is_lazy_plan()
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Replace existing column with Timestamp
    bdf = bd.from_pandas(df)
    bdf["A"] = pd.Timestamp("2024-01-1")
    pdf = df.copy()
    pdf["A"] = pd.Timestamp("2024-01-1")
    assert bdf.is_lazy_plan()
    _test_equal(bdf, pdf, check_pandas_types=False)


def test_parquet_read_partitioned(datapath):
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


def test_parquet_read_partitioned_filter(datapath):
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


def test_parquet_read_shape_head(datapath):
    """
    Test to catch a case where the original manager goes out of scope
    causing the parallel get to become invalid.
    """
    path = datapath("dataframe_library/df1.parquet")

    def bodo_impl():
        df = bd.read_parquet(path)
        return df.shape, df.head(4)

    def pd_impl():
        df = pd.read_parquet(path)
        return df.shape, df.head(4)

    bdf_shape, bdf_head = bodo_impl()
    pdf_shape, pdf_head = pd_impl()
    assert bdf_shape == pdf_shape
    _test_equal(bdf_head, pdf_head)


def test_project_after_filter(datapath):
    """Test creating a plan with a Projection on top of a filter works"""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1[bodo_df1.D > 80][["B", "A"]]

    # Make sure bodo_df2 is unevaluated at this point.
    assert bodo_df2.is_lazy_plan()

    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df2 = py_df1[py_df1.D > 80][["B", "A"]]

    # TODO: remove copy when df.apply(axis=0) is implemented
    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_merge():
    """Simple test for DataFrame merge."""
    df1 = pd.DataFrame(
        {
            "B": ["a1", "b11", "c111"],
            "E": [1.1, 2.2, 3.3],
            "A": pd.array([2, 2, 3], "Int64"),
        },
    )
    df2 = pd.DataFrame(
        {
            "Cat": pd.array([2, 3, 8], "Int64"),
            "Dog": ["a1", "b222", "c33"],
        },
    )

    bdf1 = bd.from_pandas(df1)
    bdf2 = bd.from_pandas(df2)

    df3 = df1.merge(df2, how="inner", left_on=["A"], right_on=["Cat"])
    bdf3 = bdf1.merge(bdf2, how="inner", left_on=["A"], right_on=["Cat"])
    # Make sure bdf3 is unevaluated at this point.
    assert bdf3.is_lazy_plan()

    _test_equal(
        bdf3.copy(),
        df3,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_merge_swith_side():
    """Test merge with left table smaller than right table so DuckDB reorders the input
    tables to use the smaller table as build.
    """
    df1 = pd.DataFrame(
        {
            "A": pd.array([2, 2, 3], "Int64"),
            "B": ["a1", "b11", "c111"],
        },
    )
    df2 = pd.DataFrame(
        {
            "D": ["a1", "b222", "c33"],
            "A": pd.array([2, 3, 8], "Int64"),
            "E": [1.1, 2.2, 3.3],
        },
    )
    bdf1 = bd.from_pandas(df1)
    bdf2 = bd.from_pandas(df2)
    df3 = df1.merge(df2, how="inner", on=["A"])
    bdf3 = bdf1.merge(bdf2, how="inner", on=["A"])
    # Make sure bdf3 is unevaluated at this point.
    assert bdf3.is_lazy_plan()

    _test_equal(
        bdf3.copy(),
        df3,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_dataframe_copy(index_val):
    """
    Test that creating a Pandas DataFrame from a Bodo DataFrame has the correct index.
    """
    df1 = pd.DataFrame(
        {
            "A": pd.array([2, 2, 3], "Int64"),
            "B": ["a1", "b11", "c111"],
            "E": [1.1, 2.2, 3.3],
        },
    )
    df1.index = index_val[: len(df1)]

    bdf = bd.from_pandas(df1)

    pdf_from_bodo = pd.DataFrame(bdf)

    _test_equal(df1, pdf_from_bodo, sort_output=True)


def test_dataframe_sort(datapath):
    """Very simple test for sorting for sanity checking."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1.sort_values(
        by=["D", "A"], ascending=[True, False], na_position="last"
    )

    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df2 = py_df1.sort_values(
        by=["D", "A"], ascending=[True, False], na_position="last"
    )

    assert bodo_df2.is_lazy_plan()

    _test_equal(
        bodo_df2,
        py_df2,
        check_pandas_types=False,
        sort_output=False,
        reset_index=True,
    )


def test_series_sort(datapath):
    """Very simple test for sorting for sanity checking."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1["D"]
    bodo_df3 = bodo_df2.sort_values(ascending=False, na_position="last")

    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df2 = py_df1["D"]
    py_df3 = py_df2.sort_values(ascending=False, na_position="last")

    assert bodo_df3.is_lazy_plan()

    _test_equal(
        bodo_df3,
        py_df3,
        check_pandas_types=False,
        sort_output=False,
        reset_index=True,
    )


def test_basic_groupby():
    """
    Test a simple groupby operation.
    """
    df1 = pd.DataFrame(
        {
            "B": ["a1", "b11", "c111"] * 2,
            "E": [1.1, 2.2, 13.3] * 2,
            "A": pd.array([pd.NA, 2, 3] * 2, "Int64"),
        },
        index=[0, 41, 2] * 2,
    )

    bdf1 = bd.from_pandas(df1)
    bdf2 = bdf1.groupby("A")["E"].sum()
    assert bdf2.is_lazy_plan()

    df2 = df1.groupby("A")["E"].sum()

    _test_equal(
        bdf2,
        df2,
        sort_output=True,
    )


def test_compound_projection_expression(datapath):
    """Very simple test for projection expressions."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1[(bodo_df1.A + 50) / 2 < bodo_df1.D * 2]

    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df2 = py_df1[(py_df1.A + 50) / 2 < py_df1.D * 2]

    _test_equal(
        bodo_df2,
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_projection_expression_floordiv(datapath):
    """Test for floordiv."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1[(bodo_df1.A // 3) * 7 > 15]

    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df2 = py_df1[(py_df1.A // 3) * 7 > 15]

    _test_equal(
        bodo_df2,
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_series_compound_expression(datapath):
    """Very simple test for projection expressions."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = (bodo_df1["A"] + 50) * 2 / 7

    py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    py_df2 = (py_df1["A"] + 50) * 2 / 7

    _test_equal(
        bodo_df2,
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_map_partitions():
    """Simple tests for map_partition on lazy DataFrame."""
    df = pd.DataFrame(
        {
            "E": [1.1, 2.2, 13.3] * 2,
            "A": pd.array([2, 2, 3] * 2, "Int64"),
        },
        index=[0, 41, 2] * 2,
    )

    bodo_df = bd.from_pandas(df)

    def f(df, a, b=1):
        return df.A + df.E + a + b

    bodo_df2 = bodo_df.map_partitions(f, 2, b=3)
    py_out = df.A + df.E + 2 + 3

    assert bodo_df2.is_lazy_plan()

    _test_equal(bodo_df2, py_out, check_pandas_types=False)

    # test fallback case for unsupported func
    # that returns a DataFrame
    def g(df, a, b=1):
        return df + a + b

    with pytest.warns(BodoLibFallbackWarning):
        bodo_df2 = bodo_df.map_partitions(g, 2, b=3)

    py_out = df + 2 + 3
    _test_equal(bodo_df2, py_out, check_pandas_types=False)
