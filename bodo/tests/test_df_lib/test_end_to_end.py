import datetime
import operator
import os
import tempfile
import warnings

import numba  # noqa TID253
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
import bodo.pandas as bd
from bodo.pandas.plan import (
    LogicalGetPandasReadParallel,
    LogicalGetPandasReadSeq,
    assert_executed_plan_count,
)
from bodo.pandas.utils import (
    BodoCompilationFailedWarning,
    BodoLibFallbackWarning,
    JITFallback,
)
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
        assert isinstance(bdf._mgr._plan, LogicalGetPandasReadSeq)
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
    assert isinstance(bdf._mgr._plan, LogicalGetPandasReadParallel)
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
    with assert_executed_plan_count(0):
        path = datapath("example_no_index.parquet")

        bodo_out = bd.read_parquet(path)
        py_out = pd.read_parquet(path, dtype_backend="pyarrow")

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
    with assert_executed_plan_count(0):
        path = datapath(file_path)

        bodo_out = bd.read_parquet(path)[["three", "four"]]
        py_out = pd.read_parquet(path, dtype_backend="pyarrow")[["three", "four"]]

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
                    "one": [-1.0, -4.1, 2.5, 3.0, 4.0, 6.0, 10.0],
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
        py_out = pd.read_parquet(path, dtype_backend="pyarrow")

        _test_equal(
            bodo_out,
            py_out,
        )


def test_read_parquet_len_shape(datapath):
    """Test length/shape after read parquet is correct"""
    with assert_executed_plan_count(0):
        path = datapath("example_no_index.parquet")

        bodo_out = bd.read_parquet(path)
        py_out = pd.read_parquet(path, dtype_backend="pyarrow")

        # len directly on parquet file doesn't require plan execution
        assert len(bodo_out) == len(py_out)

        # create a new lazy DF
        bodo_out2 = bd.read_parquet(path)

        # test shape: shape directly on parquet file doesn't require plan execution
        assert bodo_out2.shape == py_out.shape


def test_read_parquet_series_len_shape(datapath):
    """Test length/shape after read parquet is correct"""
    with assert_executed_plan_count(0):
        path = datapath("dataframe_library/df1.parquet")

        bodo_out = bd.read_parquet(path)
        bodo_out = bodo_out["A"]
        py_out = pd.read_parquet(path, dtype_backend="pyarrow")
        py_out = py_out["A"]

        # len directly on parquet file doesn't require plan execution
        assert len(bodo_out) == len(py_out)

        # test shape: shape directly on parquet file doesn't require plan execution
        assert bodo_out.shape == py_out.shape


@pytest.mark.jit_dependency
def test_read_parquet_filter_projection(datapath):
    """Test TPC-H Q6 bug where filter and projection pushed down to read parquet
    and filter column isn't used anywhere in the query.
    """
    with assert_executed_plan_count(0):
        path = datapath("dataframe_library/q6_sample.pq")

        def impl(lineitem):
            date1 = pd.Timestamp("1996-01-01")
            sel = (lineitem.L_SHIPDATE >= date1) & (lineitem.L_DISCOUNT >= 0.08)
            flineitem = lineitem[sel]
            return flineitem.L_EXTENDEDPRICE

        bodo_df = bd.read_parquet(path)
        bodo_df["L_SHIPDATE"] = bd.to_datetime(bodo_df.L_SHIPDATE, format="%Y-%m-%d")
        py_df = pd.read_parquet(path, dtype_backend="pyarrow")
        py_df["L_SHIPDATE"] = pd.to_datetime(py_df.L_SHIPDATE, format="%Y-%m-%d")

        bodo_out = impl(bodo_df)
        py_out = impl(py_df)

    _test_equal(
        bodo_out.copy(),
        py_out,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.jit_dependency
def test_write_parquet(index_val):
    """Test writing a DataFrame to parquet."""
    df = pd.DataFrame(
        {
            "one": [-1.0, -4.3, 2.5, 3.0, 4.0, 6.0, 10.0],
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
        py_out = pd.read_parquet(path, dtype_backend="pyarrow")
        _test_equal(
            py_out,
            df,
            check_pandas_types=False,
            sort_output=True,
            # RangeIndex order is not guaranteed
            reset_index=isinstance(index_val, pd.RangeIndex),
        )

        # Already distributed DataFrame case
        path = os.path.join(tmp, "test_write_dist.parquet")
        bodo_df = bd.from_pandas(df)

        @bodo.jit(spawn=True)
        def f(df):
            return df

        f(bodo_df)
        bodo_df.to_parquet(path)
        # Read back to check
        py_out = pd.read_parquet(path, dtype_backend="pyarrow")
        _test_equal(
            py_out,
            df,
            check_pandas_types=False,
            sort_output=True,
            # RangeIndex order is not guaranteed
            reset_index=isinstance(index_val, pd.RangeIndex),
        )


def test_projection(datapath):
    """Very simple test for projection for sanity checking."""
    bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    bodo_df2 = bodo_df1["D"]

    py_df1 = pd.read_parquet(
        datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
    )
    py_df2 = py_df1["D"]

    _test_equal(
        bodo_df2,
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        # Index is initially RangeIndex, so we ignore final order.
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
    # Make sure bodo_df2 is unevaluated in the process.
    with assert_executed_plan_count(0):
        op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]

        bodo_df1 = bd.read_parquet(datapath(file_path))
        bodo_df2 = bodo_df1[eval(f"bodo_df1.A {op_str} 20")]

    pre, post = bd.plan.getPlanStatistics(bodo_df2._mgr._plan)

    _test_equal(pre, 2)
    _test_equal(post, 1)

    with assert_executed_plan_count(0):
        py_df1 = pd.read_parquet(datapath(file_path), dtype_backend="pyarrow")
        py_df2 = py_df1[eval(f"py_df1.A {op_str} 20")]

    # TODO: remove copy when df.apply(axis=0) is implemented
    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )


@pytest.mark.jit_dependency
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
    # Make sure bodo_df2 is unevaluated in the process.
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath(file_path))
        py_df1 = pd.read_parquet(datapath(file_path), dtype_backend="pyarrow")

        @bodo.jit(spawn=True)
        def f(df):
            return df

    with assert_executed_plan_count(1):
        # Force plan to execute but keep distributed.
        f(bodo_df1)
        op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]

        bodo_df2 = bodo_df1[eval(f"bodo_df1.A {op_str} 20")]
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
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )

    # Force read parquet node to execute.
    _test_equal(
        bodo_df1.copy(),
        py_df1,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )

    # Make sure bodo_df2 is unevaluated in the process.
    with assert_executed_plan_count(0):
        op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
        bodo_df2 = bodo_df1[eval(f"bodo_df1.A {op_str} 20")]
        py_df2 = py_df1[eval(f"py_df1.A {op_str} 20")]

    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )


@pytest.mark.jit_dependency
@pytest.mark.parametrize(
    "file_path",
    [
        "dataframe_library/df1.parquet",
        "dataframe_library/df1_index.parquet",
        "dataframe_library/df1_multi_index.parquet",
    ],
)
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_filter_bound_between(datapath, file_path, mode):
    """Test for filter with filter pushdown into read parquet."""
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath(file_path))

    @bodo.jit(spawn=True)
    def f(df):
        return df

    with assert_executed_plan_count(0 if not mode else 1):
        if mode == 1:
            f(bodo_df1)
        elif mode == 2:
            bodo_df1._mgr._collect()

    # Make sure bodo_df2 is unevaluated at this point.
    with assert_executed_plan_count(0):
        bodo_df2 = bodo_df1[(bodo_df1.A > 3) & (bodo_df1.A < 8)]

    py_df1 = pd.read_parquet(datapath(file_path), dtype_backend="pyarrow")
    py_df2 = py_df1[(py_df1.A > 3) & (py_df1.A < 8)]
    assert len(py_df2) == 4

    # TODO: remove copy when df.apply(axis=0) is implemented
    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_filter_multiple1_pushdown(datapath):
    """Test for multiple filter expression."""

    # Make sure bodo_df2 is unevaluated at this point.
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = bodo_df1[((bodo_df1.A < 20) & ~(bodo_df1.D > 80))]

    py_df1 = pd.read_parquet(
        datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
    )
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
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )

    # Force read parquet node to execute.
    _test_equal(
        bodo_df1.copy(),
        py_df1,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )

    # Make sure bodo_df2 is unevaluated in this process.
    with assert_executed_plan_count(0):
        bodo_df2 = bodo_df1[((bodo_df1.A < 20) & ~(bodo_df1.D > 80))]
        py_df2 = py_df1[((py_df1.A < 20) & ~(py_df1.D > 80))]

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

    # Make sure bodo_df2 is unevaluated in this process.
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = bodo_df1[bodo_df1.B == "gamma"]

    pre, post = bd.plan.getPlanStatistics(bodo_df2._mgr._plan)

    _test_equal(pre, 2)
    _test_equal(post, 1)

    py_df1 = pd.read_parquet(
        datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
    )
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

    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )

    # Force read parquet node to execute.
    _test_equal(
        bodo_df1.copy(),
        py_df1,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )

    # Make sure bodo_df2 is unevaluated at this point.
    with assert_executed_plan_count(0):
        bodo_df2 = bodo_df1[bodo_df1.B == "gamma"]

    py_df2 = py_df1[py_df1.B == "gamma"]

    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.jit_dependency
@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.gt, operator.lt, operator.ge, operator.le]
)
def test_filter_datetime_pushdown(datapath, op):
    """Test for standalone filter."""

    # Make sure bodo_df2 is unevaluated at this point.
    with assert_executed_plan_count(0):
        op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = bodo_df1[
            eval(f"bodo_df1.F {op_str} pd.to_datetime('2025-07-17 22:39:02')")
        ]

    pre, post = bd.plan.getPlanStatistics(bodo_df2._mgr._plan)

    _test_equal(pre, 2)
    _test_equal(post, 1)

    py_df1 = pd.read_parquet(
        datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
    )
    py_df2 = py_df1[eval(f"py_df1.F {op_str} pd.to_datetime('2025-07-17 22:39:02')")]

    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.jit_dependency
@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.gt, operator.lt, operator.ge, operator.le]
)
def test_filter_datetime(datapath, op):
    """Test for standalone filter."""
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )

    # Force read parquet node to execute so the filter doesn't get pushed into the read.
    _test_equal(
        bodo_df1.copy(),
        py_df1,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )

    # Make sure bodo_df2 is unevaluated at this point.
    with assert_executed_plan_count(0):
        op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]

        bodo_df2 = bodo_df1[
            eval(f"bodo_df1.F {op_str} pd.to_datetime('2025-07-17 22:39:02')")
        ]

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

    # Make sure bodo_df2 is unevaluated at this point.
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = bodo_df1.head(3)

    pre, post = bd.plan.getPlanStatistics(bodo_df2._plan)

    _test_equal(pre, 2)
    _test_equal(post, 1)

    # Contents not guaranteed to be the same as Pandas so just check length.
    assert len(bodo_df2) == 3


def test_projection_head_pushdown(datapath):
    """Test for projection and head pushed down to read parquet."""

    # Make sure bodo_df2 is unevaluated at this point.
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = bodo_df1["D"]
        bodo_df3 = bodo_df2.head(3)

    # Contents not guaranteed to be the same as Pandas so just check length.
    assert len(bodo_df3) == 3


def test_series_head(datapath):
    """Test for Series.head() reading from Pandas."""
    # Make sure bodo_df3 is unevaluated in the process.
    with assert_executed_plan_count(1):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = bodo_df1["D"]
        bodo_df2.execute_plan()
        bodo_df3 = bodo_df2.head(3)

    # Contents not guaranteed to be the same as Pandas so just check length.
    assert len(bodo_df3) == 3


def test_head(datapath):
    """Test for head pushed down to read parquet."""

    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )

    _test_equal(
        bodo_df1.copy(),
        py_df1,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )

    # Make sure bodo_df2 is unevaluated at this point.
    with assert_executed_plan_count(0):
        bodo_df2 = bodo_df1.head(3)

    # Contents not guaranteed to be the same as Pandas so just check length.
    assert len(bodo_df2) == 3


@pytest.mark.jit_dependency
def test_apply(datapath, index_val):
    """Very simple test for df.apply() for sanity checking."""
    # Multi-Index apply are not supported by JIT
    n_execs = 1 if isinstance(index_val, pd.MultiIndex) else 0

    with assert_executed_plan_count(n_execs):
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


@pytest.mark.jit_dependency
def test_apply_str(datapath, index_val):
    """Test passing a string argument to func works."""
    with assert_executed_plan_count(0):
        df = pd.DataFrame(
            {
                "a": pd.array([1, 2, 3] * 10, "Int64"),
                "b": pd.array([4, 5, 6] * 10, "Int64"),
            },
            index=index_val[:30],
        )
        bdf = bd.from_pandas(df)
        out_pd = df.apply("sum", axis=1)
        out_bodo = bdf.apply("sum", axis=1)

    _test_equal(out_bodo, out_pd, check_pandas_types=False)


@pytest.mark.jit_dependency
def test_apply_non_jit(datapath, index_val):
    """Test unsupported UDFs fallback to Pandas execution."""
    with assert_executed_plan_count(1):
        df = pd.DataFrame(
            {
                "a": pd.array([1, 2, 3] * 10, "Int64"),
                "b": pd.array([4, 5, 6] * 10, "Int64"),
                "c": ["a", "b", "c"] * 10,
            },
            index=index_val[:30],
        )
        bdf = bd.from_pandas(df)

        def unknown_func(x):
            return x + 10

        def apply_func(row):
            return unknown_func(row.a)

        out_pd = df.apply(apply_func, axis=1)
        with pytest.warns(
            BodoCompilationFailedWarning, match="Compiling user defined function failed"
        ):
            out_bodo = bdf.apply(apply_func, axis=1)

    _test_equal(out_bodo, out_pd, check_pandas_types=False)


def test_chain_python_func(datapath, index_val):
    """Make sure chaining multiple Series functions that run in Python works"""
    with assert_executed_plan_count(0):
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
    _test_equal(out_bodo, out_pd, check_pandas_types=False)


@pytest.mark.jit_dependency
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
        return "A" if pd.isna(x) else "B"

    bdf = bd.from_pandas(df)
    out_pd = df.A.map(func, na_action=na_action)
    out_bodo = bdf.A.map(func, na_action=na_action)
    assert out_bodo.is_lazy_plan()
    _test_equal(out_bodo, out_pd, check_pandas_types=False)


@pytest.mark.jit_dependency
def test_series_map_non_jit(index_val):
    """Test non-jittable UDFs in ser.map still work."""
    df = pd.DataFrame(
        {
            "A": pd.array([None, None, 3, 7, 2] * 2, "Int64"),
            "B": [None, None, "B1", "C1", "Abc"] * 2,
            "C": pd.array([4, 5, 6, -1, 1] * 2, "Int64"),
        }
    )
    df.index = index_val[: len(df)]

    # Function with different return types,
    # technically this function isn't allowed in Python mode
    # either, but the branch is never executed due to the data
    # recieved.
    def func1(x):
        if x > 10:
            return "too-large"
        else:
            return x

    def unknown_func(x):
        return x + 10

    # Calling a function that is not known to bodo.
    def func2(x):
        return unknown_func(x)

    warn_msg = "Compiling user defined function failed "
    bdf = bd.from_pandas(df)
    with pytest.warns(BodoCompilationFailedWarning, match=warn_msg):
        bdf2 = bdf.C.map(func1)
    pdf = df.copy()
    pdf2 = pdf.C.map(func1)
    _test_equal(pdf2, bdf2, check_pandas_types=False)

    bdf = bd.from_pandas(df)
    with pytest.warns(BodoCompilationFailedWarning, match=warn_msg):
        bdf2 = bdf.C.map(func2)
    pdf = df.copy()
    pdf2 = pdf.C.map(func2)

    _test_equal(pdf2, bdf2, check_pandas_types=False)


@pytest.mark.jit_dependency
def test_set_df_column(datapath, index_val):
    """Test setting a dataframe column with a Series function of the same dataframe."""
    with assert_executed_plan_count(0):
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
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Single projection, existing column
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["B"] = bdf["B"].str.strip()
        pdf = df.copy()
        pdf["B"] = pdf["B"].str.strip()
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Multiple projections, new column
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["D"] = bdf["B"].str.strip().map(lambda x: x + "1")
        pdf = df.copy()
        pdf["D"] = pdf["B"].str.strip().map(lambda x: x + "1")
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Multiple projections, existing column
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["B"] = bdf["B"].str.strip().map(lambda x: x + "1")
        pdf = df.copy()
        pdf["B"] = pdf["B"].str.strip().map(lambda x: x + "1")
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Trivial case: set a column to existing column
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["D"] = bdf["B"]
        pdf = df.copy()
        pdf["D"] = pdf["B"]
    _test_equal(bdf, pdf, check_pandas_types=False)


def test_set_df_column_const(datapath, index_val):
    """Test setting a dataframe column with a constant value."""

    with assert_executed_plan_count(0):
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
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Two new integer columns
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf[["D", "G"]] = 111
        pdf = df.copy()
        pdf[["D", "G"]] = 111
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Replace existing column with float
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["B"] = 1.23
        pdf = df.copy()
        pdf["B"] = 1.23
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Replace existing column with string
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["C"] = "ABC"
        pdf = df.copy()
        pdf["C"] = "ABC"
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Replace existing column with Timestamp
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["A"] = pd.Timestamp("2024-01-1")
        pdf = df.copy()
        pdf["A"] = pd.Timestamp("2024-01-1")
    _test_equal(bdf, pdf, check_pandas_types=False)


def test_set_df_column_func_nested_arith(datapath, index_val):
    """Test setting a dataframe column with nested functions inside an arithmetic operation."""

    with assert_executed_plan_count(0):
        df = pd.DataFrame(
            {
                "A": [1.4, 2.1, 3.3],
                "B": ["A1", "B23", "C345"],
                "C": [1.1, 2.2, 3.3],
                "D": [True, False, True],
            }
        )
        df.index = index_val[: len(df)]

        # New column
        bdf = bd.from_pandas(df)
        bdf["E"] = bdf.B.str.lower().str.len() + 1
        pdf = df.copy()
        pdf["E"] = pdf.B.str.lower().str.len() + 1
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Existing column
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["B"] = bdf.B.str.lower().str.len() + 1
        pdf = df.copy()
        pdf["B"] = pdf.B.str.lower().str.len() + 1
    _test_equal(bdf, pdf, check_pandas_types=False)


def test_set_df_column_arith(datapath, index_val):
    """Test setting a dataframe column with a Series function of the same dataframe."""

    with assert_executed_plan_count(0):
        df = pd.DataFrame(
            {
                "A": pd.array([1, 2, 3, 7], "Int64"),
                "B": ["A1\t", "B1 ", "C1\n", "Abc\t"],
                "C": pd.array([4, 5, 6, -1], "Int64"),
            }
        )
        df.index = index_val[: len(df)]
        bdf = bd.from_pandas(df)

        # Test addition
        bdf = bd.from_pandas(df)
        bdf["D"] = bdf["A"] + 13
        pdf = df.copy()
        pdf["D"] = pdf["A"] + 13
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Test subtraction
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["D"] = bdf["A"] - 13
        pdf = df.copy()
        pdf["D"] = pdf["A"] - 13
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Test multiply
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["D"] = bdf["A"] * 13
        pdf = df.copy()
        pdf["D"] = pdf["A"] * 13
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Test division
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["D"] = bdf["A"] / 2
        pdf = df.copy()
        pdf["D"] = pdf["A"] / 2
    _test_equal(bdf, pdf, check_pandas_types=False)


def test_set_df_column_extra_proj(datapath, index_val):
    """Test setting a dataframe column with a Series function of the same dataframe to
    a dataframe that has column projections on top of the source dataframe.
    """
    with assert_executed_plan_count(0):
        df = pd.DataFrame(
            {
                "A": pd.array([1, 2, 3, 7], "Int64"),
                "B": ["A1\t", "B1 ", "C1\n", "Abc\t"],
                "C": pd.array([4, 5, 6, -1], "Int64"),
            }
        )
        df.index = index_val[: len(df)]

        # Single projection, new column
        bdf = bd.from_pandas(df)
        bdf2 = bdf[["C", "B"]]
        bdf2["D"] = bdf["A"] + bdf["C"]
        pdf = df.copy()
        pdf2 = pdf[["C", "B"]]
        pdf2["D"] = pdf["A"] + pdf["C"]
    _test_equal(bdf2, pdf2, check_pandas_types=False)

    # Multiple projections, new column
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf2 = bdf[["C", "B"]]
        bdf2["D"] = bdf["B"].str.strip().str.lower()
        pdf = df.copy()
        pdf2 = pdf[["C", "B"]]
        pdf2["D"] = pdf["B"].str.strip().str.lower()
    _test_equal(bdf2, pdf2, check_pandas_types=False)

    # Single projection, existing column in source dataframe
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf2 = bdf[["C", "B"]]
        bdf2["A"] = bdf["A"] + bdf["C"]
        pdf = df.copy()
        pdf2 = pdf[["C", "B"]]
        pdf2["A"] = pdf["A"] + pdf["C"]
    _test_equal(bdf2, pdf2, check_pandas_types=False)

    # Multiple projections, existing column in source dataframe
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf2 = bdf[["C", "B"]]
        bdf2["A"] = bdf["B"].str.strip().str.lower()
        pdf = df.copy()
        pdf2 = pdf[["C", "B"]]
        pdf2["A"] = pdf["B"].str.strip().str.lower()
    _test_equal(bdf2, pdf2, check_pandas_types=False)

    # Single projection, existing column in projected dataframe
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf2 = bdf[["C", "B"]]
        bdf2["B"] = bdf["A"] + bdf["C"]
        pdf = df.copy()
        pdf2 = pdf[["C", "B"]]
        pdf2["B"] = pdf["A"] + pdf["C"]
    _test_equal(bdf2, pdf2, check_pandas_types=False)

    # Multiple projections, existing column in projected dataframe
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf2 = bdf[["C", "B"]]
        bdf2["B"] = bdf["B"].str.strip().str.lower()
        pdf = df.copy()
        pdf2 = pdf[["C", "B"]]
        pdf2["B"] = pdf["B"].str.strip().str.lower()
    _test_equal(bdf2, pdf2, check_pandas_types=False)


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

    with assert_executed_plan_count(0):
        bodo_out = bd.read_parquet(path)
        py_out = pd.read_parquet(path, dtype_backend="pyarrow")

    # NOTE: Bodo DataFrames currently reads partitioned columns as
    # dictionary-encoded strings but Pandas reads them as categorical.
    _test_equal(
        bodo_out.copy(),
        py_out,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )


def test_parquet_read_partitioned_filter(datapath):
    """Test filter pushdown on partitioned parquet dataset."""
    path = datapath("dataframe_library/example_partitioned.parquet")

    with assert_executed_plan_count(0):
        bodo_out = bd.read_parquet(path)
        bodo_out = bodo_out[bodo_out.part == "a"]
        py_out = pd.read_parquet(path, dtype_backend="pyarrow")
        py_out = py_out[py_out.part == "a"]

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
        df = pd.read_parquet(path, dtype_backend="pyarrow")
        return df.shape, df.head(4)

    with assert_executed_plan_count(0):
        bdf_shape, bdf_head = bodo_impl()
        pdf_shape, pdf_head = pd_impl()
        assert bdf_shape == pdf_shape
    _test_equal(bdf_head, pdf_head)


def test_project_after_filter(datapath):
    """Test creating a plan with a Projection on top of a filter works"""

    # Make sure bodo_df2 is unevaluated at this point.
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = bodo_df1[bodo_df1.D > 80][["B", "A"]]
        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )
        py_df2 = py_df1[py_df1.D > 80][["B", "A"]]

    # TODO: remove copy when df.apply(axis=0) is implemented
    _test_equal(
        bodo_df2.copy(),
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_merge(how):
    """Simple test for DataFrame merge."""

    # Make sure bdf3 is unevaluated in the process.
    with assert_executed_plan_count(0):
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

        df3 = df1.merge(df2, how=how, left_on=["A"], right_on=["Cat"])
        bdf3 = bdf1.merge(bdf2, how=how, left_on=["A"], right_on=["Cat"])

    _test_equal(
        bdf3.copy(),
        df3,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_merge_cross():
    """Simple test for DataFrame merge with cross join."""
    with assert_executed_plan_count(0):
        df1 = pd.DataFrame(
            {
                "B": ["a1", "b11", "c111", "d1111"],
                "E": [1.1, 2.2, 3.3, 4.4],
                "A": pd.array([2, 2, 3, 4], "Int64"),
            },
        )
        df2 = pd.DataFrame(
            {
                "Cat": pd.array([2, 3, 8, 1], "Int64"),
                "Dog": ["a1", "b222", "c33", "d444"],
            },
        )

        bdf1 = bd.from_pandas(df1)
        bdf2 = bd.from_pandas(df2)

        df3 = df1.merge(df2, how="cross")
        bdf3 = bdf1.merge(bdf2, how="cross")

    _test_equal(
        bdf3.copy(),
        df3,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_merge_switch_side():
    """Test merge with left table smaller than right table so DuckDB reorders the input
    tables to use the smaller table as build.
    """
    # Make sure bdf3 is unevaluated at this point.
    with assert_executed_plan_count(0):
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

    _test_equal(
        bdf3.copy(),
        df3,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_merge_non_equi_cond():
    """Simple test for non-equi join conditions."""
    # Make sure bdf3 is unevaluated in the process.
    with assert_executed_plan_count(0):
        df1 = pd.DataFrame(
            {
                "B": pd.array([4, 5, 6], "Int64"),
                "E": [1.1, 2.2, 3.3],
                "A": pd.array([2, 2, 3], "Int64"),
            },
        )
        df2 = pd.DataFrame(
            {
                "Cat": pd.array([2, 3, 8], "Int64"),
                "Dog": pd.array([8, 3, 9], "Int64"),
            },
        )

        bdf1 = bd.from_pandas(df1)
        bdf2 = bd.from_pandas(df2)

        df3 = df1.merge(df2, how="inner", left_on=["A"], right_on=["Cat"])
        bdf3 = bdf1.merge(bdf2, how="inner", left_on=["A"], right_on=["Cat"])

        df4 = df3[df3.B < df3.Dog]
        bdf4 = bdf3[bdf3.B < bdf3.Dog]

    # Make sure filter node gets pushed into join.
    pre, post = bd.plan.getPlanStatistics(bdf4._mgr._plan)

    _test_equal(pre, 5)
    # The filter node gets pushed into join and then a join filter is inserted
    _test_equal(post, 5)

    _test_equal(
        bdf4.copy(),
        df4,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )

    # Make sure bdf3 is unevaluated at this point.
    with assert_executed_plan_count(0):
        df1.loc[0, "B"] = np.nan
        bdf1 = bd.from_pandas(df1)

        nan_df3 = df1.merge(df2, how="inner", left_on=["A"], right_on=["Cat"])
        nan_bdf3 = bdf1.merge(bdf2, how="inner", left_on=["A"], right_on=["Cat"])

        nan_df4 = nan_df3[nan_df3.B < nan_df3.Dog]
        nan_bdf4 = nan_bdf3[nan_bdf3.B < nan_bdf3.Dog]

    # Make sure filter node gets pushed into join.
    pre, post = bd.plan.getPlanStatistics(nan_bdf4._mgr._plan)

    _test_equal(pre, 5)
    # The filter node gets pushed into join and then a join filter is inserted
    _test_equal(post, 5)

    _test_equal(
        nan_bdf4.copy(),
        nan_df4,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_merge_output_column_to_input_map():
    """Test for a bug in join output column to input column mapping in
    TPCH Q20.
    """
    with assert_executed_plan_count(0):
        jn2 = pd.DataFrame(
            {
                "PS_PARTKEY": pd.array([1, 4, -3, 5], "Int32"),
                "PS_SUPPKEY": pd.array([7, 1, -3, 3], "Int32"),
                "L_QUANTITY": pd.array([5.0, 17.0, 2.0, 29.0], "Float64"),
            }
        )
        supplier = pd.DataFrame(
            {
                "S_SUPPKEY": pd.array([-1, 4, 2], "Int32"),
                "S_NAME": [f"Supplier#{i:09d}" for i in range(3)],
            }
        )

        def impl(jn2, supplier):
            gb = jn2.groupby(["PS_PARTKEY", "PS_SUPPKEY"], as_index=False, sort=False)[
                "L_QUANTITY"
            ].sum()
            jn3 = gb.merge(supplier, left_on="PS_SUPPKEY", right_on="S_SUPPKEY")
            return jn3[["L_QUANTITY", "S_NAME"]]

        pd_out = impl(jn2, supplier)
        bodo_out = impl(bd.from_pandas(jn2), bd.from_pandas(supplier))

    _test_equal(
        bodo_out,
        pd_out,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_dataframe_copy(index_val):
    """
    Test that creating a Pandas DataFrame from a Bodo DataFrame has the correct index.
    """
    with assert_executed_plan_count(0):
        df1 = pd.DataFrame(
            {
                "A": pd.array([2, 2, 3], "Int64"),
                "B": ["a1", "b11", "c111"],
                "E": [1.1, 2.2, 3.3],
            },
        )
        df1.index = index_val[: len(df1)]
        bdf = bd.from_pandas(df1)

    with assert_executed_plan_count(1):
        pdf_from_bodo = pd.DataFrame(bdf)

    _test_equal(df1, pdf_from_bodo, sort_output=True)


def test_dataframe_sort(datapath):
    """Very simple test for sorting for sanity checking."""
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = bodo_df1.sort_values(
            by=["D", "A"], ascending=[True, False], na_position="last"
        )

        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )
        py_df2 = py_df1.sort_values(
            by=["D", "A"], ascending=[True, False], na_position="last"
        )

    _test_equal(
        bodo_df2,
        py_df2,
        check_pandas_types=False,
        sort_output=False,
        reset_index=True,
    )


def test_series_sort(datapath):
    """Very simple test for sorting for sanity checking."""
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = bodo_df1["D"]
        bodo_df3 = bodo_df2.sort_values(ascending=False, na_position="last")

        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )
        py_df2 = py_df1["D"]
        py_df3 = py_df2.sort_values(ascending=False, na_position="last")

    _test_equal(
        bodo_df3,
        py_df3,
        check_pandas_types=False,
        sort_output=False,
        reset_index=True,
    )


@pytest.fixture(
    params=[
        pytest.param(True, id="dropna-True"),
        pytest.param(False, id="dropna-False"),
    ],
    scope="module",
)
def dropna(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(True, id="as_index-True"),
        pytest.param(False, id="as_index-False"),
    ],
    scope="module",
)
def as_index(request):
    return request.param


def test_series_groupby(dropna, as_index):
    """
    Test a simple groupby operation.
    """
    with assert_executed_plan_count(0):
        df1 = pd.DataFrame(
            {
                "B": ["a1", "b11", "c111"] * 2,
                "E": pd.array([1.1, pd.NA, 13.3, pd.NA, pd.NA, 13.3], "Float64"),
                "A": pd.array([pd.NA, 2, 3] * 2, "Int64"),
            },
            index=[0, 41, 2] * 2,
        )
        bdf1 = bd.from_pandas(df1)
        bdf2 = bdf1.groupby("A", as_index=as_index, dropna=dropna)["E"].sum()
        df2 = df1.groupby("A", as_index=as_index, dropna=dropna)["E"].sum()

    _test_equal(bdf2, df2, sort_output=True, reset_index=True, check_pandas_types=False)


@pytest.mark.parametrize(
    "selection",
    [pytest.param(None, id="select_all"), pytest.param(["C", "A"], id="select_subset")],
)
def test_dataframe_groupby(dropna, as_index, selection):
    """
    Test a simple groupby operation.
    """
    with assert_executed_plan_count(0):
        df1 = pd.DataFrame(
            {
                "A": pd.array([1, 2, pd.NA, 2147483647] * 3, "Int32"),
                "B": ["A", "B"] * 6,
                "E": [False, True] * 6,
                "D": pd.array(
                    [i * 2 if (i**2) % 3 == 0 else pd.NA for i in range(12)], "Int32"
                ),
                "C": pd.array([0.2, 0.2, 0.3] * 4, "Float32"),
            }
        )

        bdf1 = bd.from_pandas(df1)

        if selection is None:
            bdf2 = bdf1.groupby(["D", "E"], as_index=as_index, dropna=dropna).sum()
            df2 = df1.groupby(["D", "E"], as_index=as_index, dropna=dropna).sum()
        else:
            bdf2 = bdf1.groupby(["D", "E"], as_index=as_index, dropna=dropna)[
                selection
            ].sum()
            df2 = df1.groupby(["D", "E"], as_index=as_index, dropna=dropna)[
                selection
            ].sum()

    _test_equal(bdf2, df2, sort_output=True, reset_index=True)


def test_groupby_fallback():
    """Checks that fallback is properly supported for DataFrame and Series groupby
    when unsupported arguments are provided.
    """

    with assert_executed_plan_count(0):
        df = pd.DataFrame({"A": pd.array([pd.NA, 2, 1, 2], "Int32"), "B": [1, 2, 3, 4]})
        bdf = bd.from_pandas(df)

    # Series groupby
    with pytest.warns(BodoLibFallbackWarning):
        fallback_out = bdf.groupby("A", dropna=False, as_index=False, sort=True)[
            "B"
        ].sum(engine="cython")

    pandas_out = df.groupby("A", dropna=False, as_index=False, sort=True)["B"].sum(
        engine="cython"
    )
    _test_equal(pandas_out, fallback_out)

    bdf2 = bd.from_pandas(df)

    # DataFrame groupby
    with pytest.warns(BodoLibFallbackWarning):
        fallback_out = bdf2.groupby("A", dropna=False, as_index=False, sort=True).sum(
            engine="cython"
        )

    pandas_out = df.groupby("A", dropna=False, as_index=False, sort=True).sum(
        engine="cython"
    )
    _test_equal(pandas_out, fallback_out)


@pytest.fixture(scope="module")
def groupby_agg_df(request):
    return pd.DataFrame(
        {
            "A": pd.array([1, 2, pd.NA, 2147483647] * 3, "Int32"),
            "D": pd.array(
                [i * 2 if (i**2) % 3 == 0 else pd.NA for i in range(12)], "Int32"
            ),
            "B": pd.array(["A", "B", pd.NA] * 4),
            "C": pd.array([0.2, 0.2, 0.3] * 4, "Float32"),
            "T": pd.timedelta_range("1 day", periods=12, freq="D", unit="ns"),
        }
    )


@pytest.mark.parametrize(
    "func, kwargs",
    [
        pytest.param({"A": "mean", "D": "count"}, {}, id="func_dict"),
        pytest.param(["sum", "count"], {}, id="func_list"),
        pytest.param("sum", {}, id="func_str"),
        pytest.param(
            None,
            {
                "mean_A": pd.NamedAgg("A", "mean"),
                "count_D": pd.NamedAgg("D", "count"),
                "count_A": pd.NamedAgg("A", "count"),
                "sum_D": pd.NamedAgg("D", "sum"),
            },
            id="func_kwargs",
        ),
    ],
)
def test_groupby_agg(groupby_agg_df, as_index, dropna, func, kwargs):
    with assert_executed_plan_count(0):
        df1 = groupby_agg_df
        bdf1 = bd.from_pandas(df1)
        bdf2 = bdf1.groupby("B", as_index=as_index, dropna=dropna).agg(func, **kwargs)
        df2 = df1.groupby("B", as_index=as_index, dropna=dropna).agg(func, **kwargs)
    _test_equal(bdf2, df2, check_pandas_types=False, sort_output=True, reset_index=True)


@pytest.mark.parametrize(
    "func, kwargs",
    [
        pytest.param(["sum", "count"], {}, id="func_list"),
        pytest.param("sum", {}, id="func_str"),
        pytest.param(
            None,
            {"mean_A": "mean", "count_A": "count", "sum_A": "sum"},
            id="func_kwargs",
        ),
    ],
)
def test_series_groupby_agg(groupby_agg_df, as_index, dropna, func, kwargs):
    with assert_executed_plan_count(0):
        df1 = groupby_agg_df
        bdf1 = bd.from_pandas(df1)
        # Dict values plus as_index raises SpecificationError in Bodo/Pandas
        if (isinstance(func, dict) or kwargs) and as_index:
            return
        bdf2 = bdf1.groupby("B", as_index=as_index, dropna=dropna)["A"].agg(
            func, **kwargs
        )
        df2 = df1.groupby("B", as_index=as_index, dropna=dropna)["A"].agg(
            func, **kwargs
        )
    _test_equal(bdf2, df2, check_pandas_types=False, sort_output=True, reset_index=True)


@pytest.mark.parametrize(
    "func",
    [
        "sum",
        "mean",
        "count",
        "max",
        "min",
        "median",
        "nunique",
        "size",
        "var",
        "std",
        "skew",
    ],
)
def test_groupby_agg_numeric(groupby_agg_df, func):
    """Tests supported aggfuncs on simple numeric (floats and ints)."""

    bdf1 = bd.from_pandas(groupby_agg_df)

    cols = ["D", "A", "C"]

    bdf2 = getattr(bdf1.groupby("B")[cols], func)()
    df2 = getattr(groupby_agg_df.groupby("B")[cols], func)()

    assert bdf2.is_lazy_plan()

    _test_equal(bdf2, df2, sort_output=True, reset_index=True, check_pandas_types=False)


def test_size_no_val(groupby_agg_df, as_index):
    """Test groupby.size() on a DataFrame without value columns."""
    sel_cols = ["D", "A"]

    df = groupby_agg_df[sel_cols]
    bdf = bd.from_pandas(df)

    with assert_executed_plan_count(0):
        bdf2 = bdf.groupby(sel_cols, as_index=as_index).size()
    pdf = df.groupby(sel_cols, as_index=as_index).size()

    _test_equal(
        bdf2, pdf, sort_output=True, check_pandas_types=False, reset_index=~as_index
    )


@pytest.mark.parametrize(
    "func",
    [
        "count",
        "max",
        "min",
        "nunique",
        "size",
    ],
)
def test_groupby_agg_ordered(func):
    """Tests supported aggfuncs on other simple data types."""
    with assert_executed_plan_count(0):
        # string, datetime, bool
        df = pd.DataFrame(
            {
                "A": pd.array([True, pd.NA, False, True] * 3),
                "B": pd.array([pd.NA, "pq", "rs", "abc", "efg", "hij"] * 2),
                "D": pd.date_range(
                    "1988-01-01", periods=12, freq="D"
                ).to_series(),  # timestamp[ns]
                "F": pd.date_range("1988-01-01", periods=12, freq="D")
                .to_series()
                .dt.date,  # date32
                "T": pd.timedelta_range(
                    "1 day", periods=12, freq="D", unit="ns"
                ),  # duration
                "K": ["A", "A", "B"] * 4,
            }
        )

        bdf1 = bd.from_pandas(df)

        bdf2 = getattr(bdf1.groupby("K"), func)()
        df2 = getattr(df.groupby("K"), func)()

    _test_equal(bdf2, df2, sort_output=True, reset_index=True, check_pandas_types=False)


def test_compound_projection_expression(datapath):
    """Very simple test for projection expressions."""

    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = bodo_df1[(bodo_df1.A + 50) / 2 < bodo_df1.D * 2]

        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )
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
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = bodo_df1[(bodo_df1.A // 3) * 7 > 15]

        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )
        py_df2 = py_df1[(py_df1.A // 3) * 7 > 15]

    _test_equal(
        bodo_df2,
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_projection_expression_mod(datapath):
    """Test for mod."""
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = bodo_df1[(bodo_df1.A % 5) > 2]

        py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
        py_df2 = py_df1[(py_df1.A % 5) > 2]

    _test_equal(
        bodo_df2,
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_series_mod(datapath):
    """Test for mod."""
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = bodo_df1["A"] % 5

        py_df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
        py_df2 = py_df1["A"] % 5

    _test_equal(
        bodo_df2,
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_series_compound_expression(datapath):
    """Very simple test for projection expressions."""
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = (bodo_df1["A"] + 50) * 2 / 7

        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )
        py_df2 = (py_df1["A"] + 50) * 2 / 7
        py_df2 = py_df2.astype(pd.ArrowDtype(pa.float64()))

    _test_equal(
        bodo_df2,
        py_df2,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_series_arith_binops(datapath, index_val):
    """Test various cases of Series binary operations."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["aa", "bb", "c"], "C": [4, 5, 6]})
    df.index = index_val[: len(df)]

    bdf = bd.from_pandas(df)

    # Simple expression with constant
    with assert_executed_plan_count(0):
        S = df["A"] + 1
        bodo_S = bdf["A"] + 1

    _test_equal(
        bodo_S.execute_plan(),
        S,
        check_pandas_types=False,
    )

    # BodoScalar expression
    with assert_executed_plan_count(0):
        S = df["A"] + df["C"].sum()
        bodo_S = bdf["A"] + bdf["C"].sum()

    _test_equal(
        bodo_S.execute_plan(),
        S,
        check_pandas_types=False,
    )

    # BodoScalar expression on top of another expression
    with assert_executed_plan_count(0):
        S = df["A"] + 1 + df["C"].sum()
        bodo_S = bdf["A"] + 1 + bdf["C"].sum()

    _test_equal(
        bodo_S.execute_plan(),
        S,
        check_pandas_types=False,
    )


def test_series_cmp_binops(datapath, index_val):
    """Test various cases of Series binary operations."""
    df = pd.DataFrame({"A": [1, 20, 300], "B": ["aa", "bb", "c"], "C": [4, 5, 6]})
    df.index = index_val[: len(df)]

    bdf = bd.from_pandas(df)

    # Simple expression with constant
    with assert_executed_plan_count(0):
        S = df["A"] > 1
        bodo_S = bdf["A"] > 1

    _test_equal(
        bodo_S.execute_plan(),
        S,
        check_pandas_types=False,
    )

    # BodoScalar expression
    with assert_executed_plan_count(0):
        S = df["A"] > df["C"].sum()
        bodo_S = bdf["A"] > bdf["C"].sum()

    _test_equal(
        bodo_S.execute_plan(),
        S,
        check_pandas_types=False,
    )

    # BodoScalar expression on top of another expression
    with assert_executed_plan_count(0):
        S = df["A"] + 1 > df["C"].sum()
        bodo_S = bdf["A"] + 1 > bdf["C"].sum()

    _test_equal(
        bodo_S.execute_plan(),
        S,
        check_pandas_types=False,
    )


def test_scalar_arith_binops(datapath, index_val):
    """Test various cases of BodoScalar binary operations."""

    df = pd.DataFrame({"A": [1, 2, 3], "B": ["aa", "bb", "c"], "C": [4, 5, 6]})
    df.index = index_val[: len(df)]

    bdf = bd.from_pandas(df)

    # Simple expression with constant
    with assert_executed_plan_count(0):
        S = df["A"].sum() + 1
        bodo_S = bdf["A"].sum() + 1

    _test_equal(bodo_S.get_value(), S)

    # Two BodoScalar expressions
    with assert_executed_plan_count(0):
        S = df["A"].sum() + df["C"].sum()
        bodo_S = bdf["A"].sum() + bdf["C"].sum()

    _test_equal(bodo_S.get_value(), S)

    # BodoScalar/Series expressions
    with assert_executed_plan_count(0):
        S = df["A"].sum() + df["C"]
        bodo_S = bdf["A"].sum() + bdf["C"]

    # TODO[BSE-5121]: Fix crash when execute_plan/get_value is not used directly
    _test_equal(
        bodo_S.execute_plan(),
        S,
        check_pandas_types=False,
    )

    # BodoScalar non-scalar expressions
    with assert_executed_plan_count(1):
        S = df["A"].sum() + np.ones(4)
        bodo_S = bdf["A"].sum() + np.ones(4)

    _test_equal(
        bodo_S,
        S,
        check_pandas_types=False,
    )


@pytest.mark.jit_dependency
def test_map_partitions_df():
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

    with assert_executed_plan_count(1):
        bodo_df2 = bodo_df.map_partitions(f, 2, b=3)
        py_out = df.A + df.E + 2 + 3

    _test_equal(bodo_df2, py_out, check_pandas_types=False)

    with assert_executed_plan_count(2):
        # test fallback case for unsupported func
        # that returns a DataFrame
        def g(df, a, b=1):
            return df + a + b

        with pytest.warns(BodoLibFallbackWarning):
            bodo_df2 = bodo_df.map_partitions(g, 2, b=3)

        py_out = df + 2 + 3
    _test_equal(bodo_df2, py_out, check_pandas_types=False)


@pytest.mark.jit_dependency
def test_map_partitions_series():
    """Simple tests for map_partition on lazy Series."""
    series = pd.Series(
        pd.array([2, 2, 3] * 2, "Int64"),
        index=[0, 41, 2] * 2,
    )

    bodo_series = bd.from_pandas(series.to_frame(name="A")).A

    def f(series, a, b=1):
        return series + a + b

    with assert_executed_plan_count(1):
        bodo_series2 = bodo_series.map_partitions(f, 2, b=3)
        py_out = series + 2 + 3

    _test_equal(bodo_series2, py_out, check_pandas_types=False, check_names=False)

    with assert_executed_plan_count(2):
        # test fallback case for unsupported func
        # that returns a DataFrame
        def g(series, a, b=1):
            return pd.DataFrame({"A": series})

        with pytest.warns(BodoLibFallbackWarning):
            bodo_df = bodo_series.map_partitions(g, 2, b=3)

        py_out = pd.DataFrame({"A": series})
    _test_equal(bodo_df, py_out, check_pandas_types=False)


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
def test_series_filter_pushdown(datapath, file_path, op):
    """Test for series filter with filter pushdown into read parquet."""

    # Make sure bodo_filter_a is unevaluated in the process.
    with assert_executed_plan_count(0):
        op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]

        bodo_df1 = bd.read_parquet(datapath(file_path))
        bodo_series_a = bodo_df1["A"]
        bodo_filter_a = bodo_series_a[eval(f"bodo_series_a {op_str} 20")]

    pre, post = bd.plan.getPlanStatistics(bodo_filter_a._mgr._plan)

    _test_equal(pre, 3)
    _test_equal(post, 2)

    with assert_executed_plan_count(0):
        py_df1 = pd.read_parquet(datapath(file_path), dtype_backend="pyarrow")
        py_series_a = py_df1["A"]
        py_filter_a = py_series_a[eval(f"py_series_a {op_str} 20")]

    _test_equal(
        bodo_filter_a,
        py_filter_a,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )


@pytest.mark.jit_dependency
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
def test_series_filter_distributed(datapath, file_path, op):
    """Very simple test for series filter for sanity checking."""
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath(file_path))
        py_df1 = pd.read_parquet(datapath(file_path), dtype_backend="pyarrow")

        @bodo.jit(spawn=True)
        def f(df):
            return df

    with assert_executed_plan_count(1):
        # Force plan to execute but keep distributed.
        f(bodo_df1)
        op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]

    # Make sure bodo_filter_a is unevaluated in the process.
    with assert_executed_plan_count(0):
        bodo_series_a = bodo_df1["A"]
        bodo_filter_a = bodo_series_a[eval(f"bodo_series_a {op_str} 5")]

        py_series_a = py_df1["A"]
        py_filter_a = py_series_a[eval(f"py_series_a {op_str} 5")]
        assert len(py_filter_a) != 0

    _test_equal(
        bodo_filter_a,
        py_filter_a,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.jit_dependency
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
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_series_filter_series(datapath, file_path, op, mode):
    """Very simple test for series filter for sanity checking."""
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath(file_path))
        py_df1 = pd.read_parquet(datapath(file_path), dtype_backend="pyarrow")

        @bodo.jit(spawn=True)
        def f(df):
            return df

    with assert_executed_plan_count(0 if mode == 0 else 1):
        # Force plan to execute but keep distributed.
        op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
        bodo_series_a = bodo_df1["A"]
        if mode == 1:
            f(bodo_series_a)
        elif mode == 2:
            bodo_series_a._mgr._collect()

    # Make sure bodo_filter_a is unevaluated in the process.
    with assert_executed_plan_count(0):
        bodo_filter_a = bodo_series_a[eval(f"bodo_series_a {op_str} 5")]
        py_series_a = py_df1["A"]
        py_filter_a = py_series_a[eval(f"py_series_a {op_str} 5")]
        assert len(py_filter_a) != 0

    _test_equal(
        bodo_filter_a,
        py_filter_a,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_filter_source_matching():
    """Test for matching expression source dataframes in filter"""
    with assert_executed_plan_count(0):
        df = pd.DataFrame(
            {
                "A": [1.4, 2.1, 3.3],
                "B": ["A", "B", "C"],
                "C": [1.1, 2.2, 3.3],
                "D": [True, False, True],
            }
        )

        # Match series source
        bdf = bd.from_pandas(df)
        bdf2 = bdf[["B", "C", "D"]]
        bodo_out = bdf2[bdf.D]
        df2 = df[["B", "C", "D"]].copy()
        py_out = df2[df.D]
    _test_equal(
        bodo_out, py_out, check_pandas_types=False, sort_output=True, reset_index=True
    )

    # Match expression source
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf2 = bdf[["B", "C", "D"]]
        bodo_out = bdf2[(bdf.C > 2.0) & (bdf2.B != "B")]
        df2 = df[["B", "C", "D"]].copy()
        py_out = df2[(df.C > 2.0) & (df2.B != "B")]
    _test_equal(
        bodo_out, py_out, check_pandas_types=False, sort_output=True, reset_index=True
    )


def test_filter_series_isin():
    """Test dataframe filter with isin case"""
    with assert_executed_plan_count(0):
        df1 = pd.DataFrame(
            {
                "A": [1.4, 2.1, 3.3],
                "B": ["A", "B", "C"],
                "C": [1, 2, 3],
                "D": [True, False, True],
            }
        )
        df2 = pd.DataFrame(
            {
                "A": ["A", "B", "C", "D"],
                "B": [11, 2, 2, 4],
            }
        )

        bdf1 = bd.from_pandas(df1)
        bdf2 = bd.from_pandas(df2)
        bodo_out = bdf1[bdf1.C.isin(bdf2.B)]
        py_out = df1[df1.C.isin(df2.B)]

    _test_equal(
        bodo_out, py_out, check_pandas_types=False, sort_output=True, reset_index=True
    )


def test_filter_series_not_isin(index_val):
    """Test dataframe filter with not isin case"""
    with assert_executed_plan_count(0):
        df1 = pd.DataFrame(
            {
                "A": [1.4, 2.1, 3.3],
                "B": ["A", "B", "C"],
                "C": [1, 2, 3],
                "D": [True, False, True],
            },
            index=index_val[:3],
        )
        df2 = pd.DataFrame(
            {
                "A": ["A", "B", "C", "D"],
                "B": [11, 2, 2, 4],
            }
        )

        bdf1 = bd.from_pandas(df1)
        bdf2 = bd.from_pandas(df2)
        bodo_out = bdf1[~bdf1.C.isin(bdf2.B)]
        py_out = df1[~df1.C.isin(df2.B)]

        # Reverse the order so the planner flips sides to put smaller table in build
        # side creating right-anti join.
        bodo_out2 = bdf2[~bdf2.B.isin(bdf1.C)]
        py_out2 = df2[~df2.B.isin(df1.C)]

    _test_equal(
        bodo_out, py_out, check_pandas_types=False, sort_output=True, reset_index=True
    )
    _test_equal(
        bodo_out2, py_out2, check_pandas_types=False, sort_output=True, reset_index=True
    )


def test_rename(datapath, index_val):
    with assert_executed_plan_count(0):
        df = pd.DataFrame(
            {
                "a": pd.array([1, 2, 3] * 10, "Int64"),
                "b": pd.array([4, 5, 6] * 10, "Int64"),
                "c": ["a", "b", "c"] * 10,
            },
            index=index_val[:30],
        )
        bdf = bd.from_pandas(df)
        rename_dict = {"a": "alpha", "b": "bravo", "c": "charlie"}
        bdf2 = bdf.rename(columns=rename_dict)
        df2 = df.rename(columns=rename_dict)
    _test_equal(bdf2, df2, check_pandas_types=False)


@pytest.mark.jit_dependency
def test_col_set_dtypes_bug():
    """Make sure setting columns doesn't lead to failure due to inconsistent dtypes
    inside the lazy manager in sequential mode.
    """

    with temp_config_override("dataframe_library_run_parallel", False):
        df = pd.DataFrame(
            {
                "A": ["A", "B", "C"] * 2,
                "B": ["NY", "TX", "CA"] * 2,
            }
        )

        df = bd.from_pandas(df)
        df2 = df[["A", "B"]]
        df["C"] = df2.apply(lambda x: x.A + x.B, axis=1)
        print(df)


def test_topn(datapath):
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = bodo_df1.sort_values(
            by=["D", "A"], ascending=[True, False], na_position="last"
        )
        bodo_df3 = bodo_df2.head(3)

        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )
        py_df2 = py_df1.sort_values(
            by=["D", "A"], ascending=[True, False], na_position="last"
        )
        py_df3 = py_df2.head(3)

    _test_equal(
        bodo_df3,
        py_df3,
        check_pandas_types=False,
        sort_output=False,
        reset_index=True,
    )


def test_DataFrame_constructor(index_val):
    """Test creating a BodoDataFrame using regular constructor"""
    with assert_executed_plan_count(0):
        df = pd.DataFrame(
            {
                "a": pd.array([1, 2, 3] * 10, "Int64"),
                "b": pd.array([4, 5, 6] * 10, "Int64"),
                "c": ["a", "b", "c"] * 10,
            },
            index=index_val[:30],
        )
        bdf = bd.DataFrame(
            {
                "a": pd.array([1, 2, 3] * 10, "Int64"),
                "b": pd.array([4, 5, 6] * 10, "Int64"),
                "c": ["a", "b", "c"] * 10,
            },
            index=index_val[:30],
        )

    _test_equal(df, bdf, check_pandas_types=False)


def test_Series_constructor(index_val):
    """Test creating a BodoSeries using regular constructor"""
    with assert_executed_plan_count(0):
        pd_S = pd.Series(pd.array([1, 2, 3] * 10, "Int64"), index=index_val[:30])
        bodo_S = bd.Series(pd.array([1, 2, 3] * 10, "Int64"), index=index_val[:30])

    _test_equal(pd_S, bodo_S, check_pandas_types=False)


def test_series_min_max():
    """Basic test for Series min and max."""
    # Large number to ensure multiple batches
    n = 10000

    df = pd.DataFrame(
        {
            "A": np.arange(n),
            "B": np.flip(np.arange(n, dtype=np.int32)),
            "C": np.append(np.arange(n // 2), np.flip(np.arange(n // 2))),
            "C2": np.append(np.arange(n // 2) + 1.1, np.flip(np.arange(n // 2)) + 2.2),
            "D": np.append(np.flip(np.arange(n // 2)), np.arange(n // 2)),
            "E": pd.date_range("1988-01-01", periods=n, freq="D").to_series(),
            "F": pd.date_range("1988-01-01", periods=n, freq="D").to_series().dt.date,
            "G": ["a", "abc", "bc3", "d4e5f"] * (n // 4),
            "H": pd.array(
                [-1.1, 2.3, 3.4, 5.2] * (n // 4),
                dtype=pd.ArrowDtype(pa.decimal128(10, 4)),
            ),
        },
    )
    bdf = bd.from_pandas(df)
    for c in df.columns:
        bodo_min = bdf[c].min()
        bodo_max = bdf[c].max()
        py_min = df[c].min()
        py_max = df[c].max()
        with assert_executed_plan_count(2):
            assert bodo_min == py_min
            assert bodo_max == py_max


def test_series_min_max_unsupported_types():
    with assert_executed_plan_count(2):
        df = pd.DataFrame(
            {"A": pd.timedelta_range("1 day", periods=10, freq="D", unit="ns")}
        )
        bdf = bd.from_pandas(df)

        with pytest.warns(BodoLibFallbackWarning):
            bdf["A"].min()

        with pytest.warns(BodoLibFallbackWarning):
            bdf["A"].max()


@pytest.mark.parametrize("method", ["sum", "product", "count", "mean", "std"])
def test_series_reductions(method):
    """Basic test for Series sum, product, count, and mean."""
    n = 10000
    df = pd.DataFrame(
        {
            "A": np.arange(n),
            "B": np.flip(np.arange(n, dtype=np.int32)),
            "C": np.append(np.arange(n // 2), np.flip(np.arange(n // 2))),
            "C2": np.append(np.arange(n // 2) + 1.1, np.flip(np.arange(n // 2)) + 2.2),
            "D": np.append(np.flip(np.arange(n // 2)), np.arange(n // 2)),
            "E": [None] * n,
            "F": np.append(np.arange(n - 1), [None]),
        }
    )

    bdf = bd.from_pandas(df)

    for c in df.columns:
        out_pandas = getattr(df[c], method)()
        out_bodo = getattr(bdf[c], method)()
        with assert_executed_plan_count(1 if c != "E" else 0):
            assert (
                np.isclose(np.array([out_pandas]), np.array([out_bodo]), rtol=1e-6)
                if not pd.isna(out_bodo)
                else pd.isna(out_pandas)
            )


@pytest.mark.jit_dependency
def test_read_csv(datapath):
    """Very simple test to read a parquet file for sanity checking."""
    with assert_executed_plan_count(0):
        path = datapath("example.csv")
        data1_path = datapath("csv_data1.csv")
        date_path = datapath("csv_data_date1.csv")

        bodo_out = bd.read_csv(path)[["one", "four"]]
        py_out = pd.read_csv(path, dtype_backend="pyarrow")[["one", "four"]]

    _test_equal(
        bodo_out,
        py_out,
    )

    with assert_executed_plan_count(0):
        bodo_out = bd.read_csv(path, usecols=[0, 3])
        py_out = pd.read_csv(path, usecols=[0, 3], dtype_backend="pyarrow")

    _test_equal(
        bodo_out,
        py_out,
    )

    with assert_executed_plan_count(0):
        col_names = ["int0", "float0", "float1", "int1"]
        bodo_out = bd.read_csv(data1_path, names=col_names)
        py_out = pd.read_csv(data1_path, names=col_names, dtype_backend="pyarrow")

    _test_equal(
        bodo_out,
        py_out,
    )

    with assert_executed_plan_count(0):
        col_names = ["int0", "float0", "date0", "int1"]
        bodo_out = bd.read_csv(date_path, names=col_names, parse_dates=[2])
        py_out = pd.read_csv(
            date_path, names=col_names, parse_dates=[2], dtype_backend="pyarrow"
        )

    _test_equal(
        bodo_out,
        py_out,
    )


@pytest.mark.jit_dependency
def test_df_state_change():
    """Make sure dataframe state change doesn't lead to stale result id in plan
    execution"""

    with assert_executed_plan_count(0):

        @bodo.jit(spawn=True)
        def get_df(df):
            return df

        bdf = get_df(pd.DataFrame({"A": [1, 2, 3, 4, 5, 6]}))
        bdf2 = bdf.A.map(lambda x: x)

    # Collect the df, original result id is stale
    print(bdf)

    # Plan execution shouldn't fail due to stale res id
    print(bdf2)


def test_dataframe_concat(datapath):
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))[
            ["A", "D"]
        ]
        bodo_df2 = bd.read_parquet(datapath("dataframe_library/df2.parquet"))[
            ["A", "E"]
        ]
        bodo_df3 = bd.concat([bodo_df1, bodo_df2, bodo_df2])

        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )[["A", "D"]]
        py_df2 = pd.read_parquet(
            datapath("dataframe_library/df2.parquet"), dtype_backend="pyarrow"
        )[["A", "E"]]
        py_df3 = pd.concat([py_df1, py_df2, py_df2])

    _test_equal(
        bodo_df3,
        py_df3,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_series_concat(datapath):
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))["A"]
        bodo_df2 = bd.read_parquet(datapath("dataframe_library/df2.parquet"))["A"]
        bodo_df3 = bd.concat([bodo_df1, bodo_df2, bodo_df2])

        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )["A"]
        py_df2 = pd.read_parquet(
            datapath("dataframe_library/df2.parquet"), dtype_backend="pyarrow"
        )["A"]
        py_df3 = pd.concat([py_df1, py_df2, py_df2])

    _test_equal(
        bodo_df3,
        py_df3,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_isin(datapath):
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        bodo_df2 = bd.read_parquet(datapath("dataframe_library/df2.parquet"))
        bodo_df3 = (bodo_df1["D"] + 100).isin(bodo_df2["E"])

        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )
        py_df2 = pd.read_parquet(
            datapath("dataframe_library/df2.parquet"), dtype_backend="pyarrow"
        )
        py_df3 = (py_df1["D"] + 100).isin(py_df2["E"])

    _test_equal(
        bodo_df3,
        py_df3,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_drop(datapath):
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet")).drop(
            columns=["A", "F"]
        )
        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        ).drop(columns=["A", "F"])

    _test_equal(
        bodo_df1,
        py_df1,
        check_pandas_types=False,
        sort_output=False,
        reset_index=False,
    )


def test_loc(datapath):
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet")).loc[
            :, ["A", "F"]
        ]
        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        ).loc[:, ["A", "F"]]

    _test_equal(
        bodo_df1,
        py_df1,
        check_pandas_types=False,
        sort_output=False,
        reset_index=False,
    )

    # Tuple column selection
    with assert_executed_plan_count(0):
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet")).loc[
            :, ("A", "F")
        ]
        py_df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        ).loc[:, ("A", "F")]

    _test_equal(
        bodo_df1,
        py_df1,
        check_pandas_types=False,
        sort_output=False,
        reset_index=False,
    )


@pytest.mark.parametrize(
    "percentiles",
    [None, (0.1, 0.4, 0.7, 0.9), [0, 1]],
)
def test_series_describe_numeric(percentiles):
    """Test for Series describe, using approximate bounds for quantiles."""

    def kll_error_bounds(q, k=200, pmf=False):
        eps = 1.0 / np.sqrt(k) * (1.7 if pmf else 1.33)
        return max(0.0, q - eps), min(1.0, q + eps)

    n = 10000
    df = pd.DataFrame(
        {
            "A": np.arange(n),
            "B": np.flip(np.arange(n, dtype=np.int32)),
            "C": np.append(np.arange(n // 2), np.flip(np.arange(n // 2))),
            "D": np.append(np.flip(np.arange(n // 2)), np.arange(n // 2)),
            "E": list(range(n - 1)) + [None],
        }
    )

    bdf = bd.from_pandas(df)
    for c in df.columns:
        with assert_executed_plan_count(
            0 if pa.types.is_null(bdf[c].dtype.pyarrow_dtype) else 5
        ):
            describe_pd = df[c].describe(percentiles=percentiles)
            describe_bodo = bdf[c].describe(percentiles=percentiles)

        # For quantile columns, check approximate bounds instead of strict equality
        # Iterate from idx=4 to second-to-last element, which is the quantile portion.
        for q in describe_pd.index[4:-1:1]:
            approx = describe_bodo.loc[q]
            true_vals = sorted(x for x in df[c].dropna().values.tolist())
            if not true_vals:
                continue
            nvals = len(true_vals)
            float_q = (
                int(q[:-1])
            ) / 100  # Convert percentile to float, i.e. "20%" to 0.2
            lo, hi = kll_error_bounds(float_q, k=200, pmf=False)
            lo_idx = int(np.floor(lo * (nvals - 1)))
            hi_idx = int(np.ceil(hi * (nvals - 1)))
            true_low = true_vals[lo_idx]
            true_high = true_vals[hi_idx]
            assert true_low <= approx <= true_high, (
                f"{c} quantile {float_q} estimate {approx} "
                f"not within [{true_low}, {true_high}]"
            )

        # For all other stats (count, mean, std, min, max), keep exact check
        _test_equal(
            describe_bodo.reindex(index=["count", "mean", "std", "min", "max"]),
            describe_pd.reindex(index=["count", "mean", "std", "min", "max"]).astype(
                pd.ArrowDtype(pa.float64())
            ),
            check_pandas_types=False,
        )


def test_series_describe_nonnumeric():
    """Basic test for Series describe with string data."""
    df = pd.DataFrame(
        {
            "A": ["apple", "banana", "apple", "cherry", "banana", "apple"],
            "B": ["apple"] * 3 + ["APPLE"] * 2 + [None],
        }
    )

    bdf = bd.from_pandas(df)
    for c in df.columns:
        # Since BodoSeries cannot have mixed dtypes, BodoSeries.describe casts all elements to string.
        # Applying map(str) to pandas output is a workaround to enable_test_equal to compare values of differing dtypes.
        describe_pd = df[c].describe().map(str)
        describe_bodo = bdf[c].describe()
        _test_equal(describe_bodo, describe_pd, check_pandas_types=False)


@pytest.mark.jit_dependency
def test_series_describe_empty():
    """Basic test for Series describe with empty data."""

    pds = pd.Series([None] * 10)
    bds = bd.Series([None] * 10)

    with assert_executed_plan_count(0):
        # Since BodoSeries cannot have mixed dtypes, BodoSeries.describe casts all elements to string.
        # Mapping the conversion logic below avoids _test_equal() evaluating to False for correct
        # results due to "nan" != "<NA>"
        describe_bodo = bds.describe().map(
            lambda x: str(x) if not pd.isna(x) else "None"
        )
        describe_pd = pds.describe().map(lambda x: str(x) if not x else "None")
    _test_equal(describe_bodo, describe_pd, check_pandas_types=False, check_names=False)

    pds = pd.Series([1, 2, 3])
    pds = pds[pds > 4]
    bds = bd.Series([1, 2, 3])
    bds = bds[bds > 4]

    with assert_executed_plan_count(4):
        describe_pd = pds.describe()
        describe_bodo = bds.describe()
    _test_equal(describe_bodo, describe_pd, check_pandas_types=False, check_names=False)


def test_groupby_getattr_fallback_behavior():
    import warnings

    df = pd.DataFrame({"apply": [1], "B": [1], "C": [2]})
    bdf = bd.from_pandas(df)

    grouped = bdf.groupby("B")

    # Accessing a column: should not raise a warning
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _ = grouped.B
    assert not record, f"Unexpected warning when accessing column: {record}"

    # Accessing an implemented Pandas GroupBy method: should raise fallback warning
    with pytest.warns(BodoLibFallbackWarning) as record:
        _ = grouped.transform
    assert len(record) == 1

    # Accessing unknown attribute: should raise AttributeError
    with pytest.raises(AttributeError):
        _ = grouped.not_a_column


def test_series_agg():
    import pandas as pd

    import bodo.pandas as bd

    df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    bdf = bd.from_pandas(df)

    bodo_out = bdf.A.aggregate("sum")
    pd_out = df.A.aggregate("sum")
    assert bodo_out == pd_out

    bodo_out = bdf.A.aggregate(["min", "max", "count", "product"])
    pd_out = df.A.aggregate(["min", "max", "count", "product"])
    _test_equal(bodo_out, pd_out, check_pandas_types=False)


@pytest.mark.jit_dependency
def test_groupby_apply():
    """Test for a groupby.apply from TPCH Q8."""

    df = pd.DataFrame(
        {
            "A": pd.array([1, 2] * 12, "Int32"),
            "B": pd.array([1, 2, 2, 1] * 6, "Int32"),
            "C": pd.array(list(range(24)), "Int32"),
        }
    )

    def impl(df):
        def udf(df):
            denom = df["C"].sum()
            df = df[df["B"] == 2]
            num = df["C"].sum()
            return num / denom

        ret = df.groupby("A", as_index=False).apply(udf)
        ret.columns = ["A", "Q"]
        return ret

    pd_out = impl(df)
    with assert_executed_plan_count(0):
        bodo_out = impl(bd.from_pandas(df))

    _test_equal(
        bodo_out,
        pd_out,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_empty_duckdb_filter():
    """Test for when duckdb generates an empty filter."""

    lineitem = pd.DataFrame(
        {
            "L_QUANTITY": pd.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5], "Int32"),
            "L_PARTKEY": pd.array([0, 0, 2, 0, 0, 2, 0, 0, 2, 0], "Int32"),
        }
    )

    part = pd.DataFrame(
        {
            "P_PARTKEY": pd.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "Int32"),
            "P_BRAND": pd.array([0, 0, 1, 2, 3, 0, 0, 1, 2, 3], "Int32"),
        }
    )

    def impl(lineitem, part):
        jn = lineitem.merge(part, left_on="L_PARTKEY", right_on="P_PARTKEY")
        jnsel = (jn.P_BRAND == 0) & (jn.L_QUANTITY >= 0) | (jn.P_BRAND == 2)
        return jn[jnsel]

    pd_out = impl(lineitem, part)
    bodo_out = impl(bd.from_pandas(lineitem), bd.from_pandas(part))
    assert bodo_out.is_lazy_plan()

    _test_equal(
        bodo_out,
        pd_out,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )


def test_empty_aggregate_batches():
    """Test for when duckdb generates an empty filter."""

    lineitem = pd.DataFrame(
        {
            "L_QUANTITY": pd.array(list(range(12000)), "Int32"),
            "L_PARTKEY": pd.array([0, 1, 2, 0, 6, 2, 0, 8, 2, 0] * 1200, "Int32"),
            "L_EXTENDEDPRICE": pd.array(
                [5, 1, 2, 7, 6, 2, 9, 8, 2, 11] * 1200, "Float64"
            ),
            "L_DISCOUNT": pd.array(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 1200, "Float64"
            ),
        }
    )

    part = pd.DataFrame(
        {
            "P_PARTKEY": pd.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "Int32"),
            "P_BRAND": pd.array([0, 0, 1, 2, 3, 0, 0, 1, 2, 3], "Int32"),
        }
    )

    quantity_ranges = [(-1, -1), (0, 12000), (0, 2000), (2000, 4000), (9000, 12000)]
    for test_range in quantity_ranges:

        def impl(lineitem, part, test_range):
            flineitem = lineitem[
                (lineitem.L_QUANTITY >= test_range[0])
                & (lineitem.L_QUANTITY < test_range[1])
            ]
            jn = flineitem.merge(part, left_on="L_PARTKEY", right_on="P_PARTKEY")
            jn["TMP"] = jn.L_EXTENDEDPRICE * (1.0 - jn.L_DISCOUNT)
            return jn.TMP.sum()

        pd_out = impl(lineitem, part, test_range)
        bodo_out = impl(bd.from_pandas(lineitem), bd.from_pandas(part), test_range)
        assert np.isclose(bodo_out, pd_out, rtol=1e-6)


def test_set_df_column_non_arith_binops():
    """Test setting dataframe columns using BodoSeries non-arithmetic binary operations."""

    with assert_executed_plan_count(0):
        df = pd.DataFrame(
            {
                "A": ["a", "b", "c", "d"],
                "B": pd.date_range("2020-01-01", periods=4),  # datetime64[ns]
                "C": pd.timedelta_range(
                    "1 day", periods=4, unit="ns"
                ),  # timedelta64[ns]
            }
        )

        # String Series + String
        bdf = bd.from_pandas(df)
        bdf["D"] = bdf["A"] + "_suffix"
        pdf = df.copy()
        pdf["D"] = pdf["A"] + "_suffix"
    _test_equal(bdf, pdf)

    # String Series + String Series
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bodo_out = bdf["A"] + bdf["A"]
        pdf = df.copy()
        pd_out = pdf["A"] + pdf["A"]
    _test_equal(bodo_out, pd_out, check_pandas_types=False)

    # Datetime Series + DateOffset
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["D"] = bdf["B"] + pd.DateOffset(
            years=+25,
            months=+5,
            days=+12,
            hours=+8,
            minutes=+54,
            seconds=+47,
            microseconds=+282310,
        )
        pdf = df.copy()
        pdf["D"] = pdf["B"] + pd.DateOffset(
            years=+25,
            months=+5,
            days=+12,
            hours=+8,
            minutes=+54,
            seconds=+47,
            microseconds=+282310,
        )
    _test_equal(bdf, pdf)

    # Timedelta Series + Timedelta
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["D"] = bdf["C"] + pd.Timedelta(1, "d")
        pdf = df.copy()
        pdf["D"] = pdf["C"] + pd.Timedelta(1, "d")
    _test_equal(bdf, pdf)

    # Datetime Series + Timedelta
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["D"] = bdf["B"] + pd.Timedelta(1, "d")
        pdf = df.copy()
        pdf["D"] = pdf["B"] + pd.Timedelta(1, "d")
    _test_equal(bdf, pdf)

    # Datetime Series + datetime.timedelta
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["D"] = bdf["B"] + datetime.timedelta(days=2)
        pdf = df.copy()
        pdf["D"] = pdf["B"] + datetime.timedelta(days=2)
    _test_equal(bdf, pdf)

    # Timedelta Series + datetime.timedelta
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["D"] = bdf["C"] + datetime.timedelta(hours=12)
        pdf = df.copy()
        pdf["D"] = pdf["C"] + datetime.timedelta(hours=12)
    _test_equal(bdf, pdf)

    # String Series + NumPy string scalar
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df)
        bdf["D"] = bdf["A"] + np.str_("foo")
        pdf = df.copy()
        pdf["D"] = pdf["A"] + np.str_("foo")
    _test_equal(bdf, pdf)


def test_fallback_wrapper_deep_fallback():
    s = bd.Series(pd.date_range("20130101 09:10:12", periods=10, freq="MS"))

    month_end = pd.offsets.MonthEnd()
    month_end_series = pd.Series([month_end] * 10)
    with pytest.warns(BodoLibFallbackWarning) as record:
        _ = s + month_end_series

    fallback_warnings = [
        w for w in record if issubclass(w.category, BodoLibFallbackWarning)
    ]
    assert len(fallback_warnings) == 2

    warning_msg = str(fallback_warnings[1].message)
    assert "TypeError triggering deeper fallback" in warning_msg, (
        f"Unexpected warning message: {warning_msg}"
    )


def test_drop_duplicates(index_val):
    """Test for drop_duplicates API."""

    with assert_executed_plan_count(0):
        df = pd.DataFrame(
            {
                "A": pd.array([0, 1] * 50, "Int32"),
                "B": pd.array([2, 3, 4, 5] * 25, "Float64"),
            }
        )
        # TODO[BSE-5201]: Add groupby dropna flag to DuckDB's aggregate node so the flag
        # is preserved when the exprs are optimized out.
        # df.loc[99, "B"] = np.nan
        df.index = index_val[:100]
        bdf = bd.from_pandas(df).drop_duplicates()
        pdf = df.copy().drop_duplicates()
    _test_equal(
        bdf,
        pdf,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="default"),
        pytest.param({"keep": "first"}, id="keep_first"),
        pytest.param({"keep": "last"}, id="keep_last"),
    ],
)
def test_drop_duplicates_subset(kwargs):
    """Test for drop_duplicates subset API."""

    dim0 = 4
    dim1 = 6

    with assert_executed_plan_count(0):
        df = pd.DataFrame(
            {
                "A": pd.array(list(range(dim0)) * dim1 * 10, "Int32"),
                "B": pd.array(list(range(dim0, dim0 + dim1)) * dim0 * 10, "Float64"),
                "C": pd.array(list(range(dim0 * dim1 * 10)), "Float64"),
            }
        )
        pdf = df.copy().drop_duplicates(subset=["A", "B"], **kwargs)[["A", "B"]]
        bdf = bd.from_pandas(df).drop_duplicates(subset=["A", "B"], **kwargs)[
            ["A", "B"]
        ]
    _test_equal(
        bdf,
        pdf,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.jit_dependency
def test_uncompilable_map():
    """Test for maps that can't be compiled."""

    with assert_executed_plan_count(1):
        df = pd.DataFrame(
            {
                "A": pd.array([0, 1] * 100, "Int32"),
            }
        )

        def uncompilable(x):
            # Numba can't compile functions containing imports.
            import operator

            operator.add
            return x

        bdf = bd.from_pandas(df)
        pdf = df.copy()

        bdf["B"] = bdf["A"].map(uncompilable)
        pdf["B"] = pdf["A"].map(uncompilable)
    _test_equal(
        bdf,
        pdf,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )


@pytest.mark.jit_dependency
def test_numba_map():
    """Test for maps with already jit annotated functions."""

    with assert_executed_plan_count(0):
        df = pd.DataFrame(
            {
                "A": pd.array([0, 1] * 100, "Int32"),
            }
        )

        @numba.njit
        def already_compiled(x):
            return x

        bdf = bd.from_pandas(df)
        pdf = df.copy()

        bdf["B"] = bdf["A"].map(already_compiled)
        pdf["B"] = pdf["A"].map(already_compiled)
    _test_equal(
        bdf,
        pdf,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )


def test_df_reset_index():
    """Test for DataFrame reset_index API."""

    with assert_executed_plan_count(0):
        df = pd.DataFrame(
            {"A": [1, 2, 3]}, index=pd.Index(["a", "b", "c"], name="Chris")
        )
        bdf = bd.from_pandas(df).reset_index()
        pdf = df.reset_index()
    _test_equal(
        bdf,
        pdf,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df).reset_index(drop=True)
        pdf = df.reset_index(drop=True)
    _test_equal(
        bdf,
        pdf,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )
    with assert_executed_plan_count(0):
        multi_array = [[1, 1, 2, 2], ["red", "blue", "red", "blue"]]
        multi_idx = pd.MultiIndex.from_arrays(multi_array)
        df = pd.DataFrame({"A": [1, 2, 3, 4]}, index=multi_idx)
        bdf = bd.from_pandas(df).reset_index()
        pdf = df.reset_index()
    _test_equal(
        bdf,
        pdf,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )
    with assert_executed_plan_count(0):
        bdf = bd.from_pandas(df).reset_index(names=["numbers", "colors"])
        pdf = df.reset_index(names=["numbers", "colors"])
    _test_equal(
        bdf,
        pdf,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )
    with assert_executed_plan_count(0):
        long_array = [
            [0, 0, 0, 0],
            [1, 1, 2, 2],
            ["red", "blue", "red", "blue"],
            ["Pitt", "Pitt", "CMU", "CMU"],
        ]
        long_index = pd.MultiIndex.from_arrays(
            long_array, names=["Rank", "B", "A", "School"]
        )
        pds = pd.DataFrame({"C": [1, 2, 3, 4]}, index=long_index)
        bd.from_pandas(pds).reset_index(level=[0, 1])
        pds = pds.reset_index(level=[0, 1])
    _test_equal(
        bdf,
        pdf,
        check_pandas_types=False,
        sort_output=True,
        reset_index=False,
    )


def test_series_reset_index():
    """Test for Series reset_index API."""

    # Tests basic Series.reset_index
    with assert_executed_plan_count(0):
        s = pd.Series([1, 2, 3, 4], index=["A", "B", "C", "D"], name="Bodo")
        bds = bd.Series(s).reset_index()
        pds = s.reset_index()
    _test_equal(
        bds,
        pds,
        check_pandas_types=False,
        reset_index=False,
    )
    # Tests basic Series.reset_index with name arg
    with assert_executed_plan_count(0):
        bds = bd.Series(s).reset_index(name="Inc")
        pds = s.reset_index(name="Inc")
    _test_equal(
        bds,
        pds,
        check_pandas_types=False,
        reset_index=False,
    )
    # Tests Series.reset_index with MultiIndex
    with assert_executed_plan_count(0):
        multi_array = [[1, 1, 2, 2], ["red", "blue", "red", "blue"]]
        multi_idx = pd.MultiIndex.from_arrays(multi_array)
        s = pd.Series([1, 2, 3, 4], index=multi_idx, name="Pitt")
        bds = bd.Series(s).reset_index(name="Penn")
        pds = s.reset_index(name="Penn")
    _test_equal(
        bds,
        pds,
        check_pandas_types=False,
        reset_index=False,
    )
    # Tests Series.reset_index with drop=True
    with assert_executed_plan_count(0):
        bds = bd.Series(s).reset_index(drop=True)
        pds = s.reset_index(drop=True)
    _test_equal(
        bds,
        pds,
        check_pandas_types=False,
        reset_index=False,
    )
    # Tests level argument
    with assert_executed_plan_count(0):
        long_array = [
            [0, 0, 0, 0],
            [1, 1, 2, 2],
            ["red", "blue", "red", "blue"],
            ["Pitt", "Pitt", "CMU", "CMU"],
        ]
        long_index = pd.MultiIndex.from_arrays(
            long_array, names=["Rank", "B", "A", "School"]
        )
        pds = pd.Series([1, 2, 3, 4], index=long_index, name="Happy")
        bds = bd.Series(pds).reset_index(level=[0, 1])
        pds = pds.reset_index(level=[0, 1])
    _test_equal(
        bds,
        pds,
        check_pandas_types=False,
        reset_index=False,
    )


@pytest.mark.jit_dependency
def test_series_reset_index_compute():
    """Test Series.reset_index in between computation."""

    df = pd.DataFrame(
        {
            "City": ["Pittsburgh", "Boston", "New York", "Seattle"],
            "Score": [92, 85, 88, 90],
        }
    ).set_index("City")

    with assert_executed_plan_count(0):
        s = df["Score"]
        bds = bd.Series(s)
        s = s.map(lambda x: x * 3)
        bds = bds.map(lambda x: x * 3)
        bds = bds.reset_index()
        bds["Decremented"] = bds["Score"] - 20
        pds = s.reset_index()
        pds["Decremented"] = pds["Score"] - 20

    _test_equal(
        bds,
        pds,
        check_pandas_types=False,
        reset_index=False,
    )


@pytest.mark.jit_dependency
def test_series_reset_index_pipeline():
    """Test reading from CSV, groupby + sum, reset_index, and more computes."""

    df = pd.DataFrame(
        {
            "category": ["A", "B", "A", "C", "B", "A", "C"],
            "value": [10, 20, 10, 30, 40, 50, 60],
            "timestamp": pd.date_range(start="2025-01-01", periods=7, freq="D"),
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "sample_data.csv")

        df.to_csv(csv_path, index=False)

        with assert_executed_plan_count(0):
            bdf = bd.read_csv(csv_path)
            pdf = pd.read_csv(csv_path, dtype_backend="pyarrow")
            bds = bdf.groupby("category")["value"].sum().reset_index()
            pds = pdf.groupby("category")["value"].sum().reset_index()

    _test_equal(
        bds,
        pds,
        check_pandas_types=False,
        reset_index=True,
        sort_output=True,
    )

    long_array = [
        [1, 1, 2],
        ["red", "blue", "green"],
    ]

    long_index = pd.MultiIndex.from_arrays(long_array, names=["Number", "Color"])

    with assert_executed_plan_count(0):
        bds.index = long_index
        pds.index = long_index

        bds = bds.map(lambda x: x * 3)
        pds = pds.map(lambda x: x * 3)

        bds["tip"] = bds["value"] / 10 + 5
        pds["tip"] = pds["value"] / 10 + 5

        bds = bds.reset_index(level=[0])
        pds = pds.reset_index(level=[0])

    _test_equal(
        bds,
        pds,
        check_pandas_types=False,
        reset_index=True,
        sort_output=True,
    )


@pytest.mark.jit_dependency
def test_dataframe_reset_index_pipeline():
    """Test reading CSV, setting MultiIndex, resetting index, and computing."""

    df = pd.DataFrame(
        {
            "Number": [1, 1, 2, 2],
            "Color": ["red", "blue", "red", "blue"],
            "value": [10, 20, 30, 40],
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "multiindex_data.csv")
        df.to_csv(csv_path, index=False)

        with assert_executed_plan_count(0):
            bdf = bd.read_csv(csv_path)
            pdf = pd.read_csv(csv_path, dtype_backend="pyarrow")

            bdf = bdf.set_index(["Number", "Color"])
            pdf = pdf.set_index(["Number", "Color"])

            bdf["double"] = bdf["value"] * 2
            pdf["double"] = pdf["value"] * 2

            bdf = bdf.reset_index(level="Color")
            pdf = pdf.reset_index(level="Color")

    _test_equal(
        bdf,
        pdf,
        check_pandas_types=False,
        reset_index=False,
    )


def test_map_with_state():
    def init_state():
        return {1: 7}

    def per_row(state, row):
        return "bodo" + str(row + state[1])

    a = pd.Series(list(range(20)))
    ba = bd.Series(a)
    res = a.map(lambda x: "bodo" + str(x + 7))
    with assert_executed_plan_count(1):
        bres = ba.map_with_state(init_state, per_row)

    _test_equal(
        bres,
        res,
        check_pandas_types=False,
        reset_index=False,
        check_names=False,
    )

    with assert_executed_plan_count(0):
        bres = ba.map_with_state(
            init_state, per_row, output_type=pd.Series(dtype="string[pyarrow]")
        )

    _test_equal(
        bres,
        res,
        check_pandas_types=False,
        reset_index=False,
        check_names=False,
    )


def test_map_partitions_with_state():
    class mystate:
        def __init__(self):
            self.dict = {1: 5}

    def init_state():
        return mystate()

    def per_batch(state, batch, *args, **kwargs):
        def per_row(row):
            return "bodo" + str(row + state.dict[1] + args[0] + kwargs["bodo"])

        return batch.map(per_row)

    a = pd.Series(list(range(20)))
    ba = bd.Series(a)
    res = a.map(lambda x: "bodo" + str(x + 7))
    with assert_executed_plan_count(1):
        bres = ba.map_partitions_with_state(init_state, per_batch, 1, bodo=1)

    _test_equal(
        bres,
        res,
        check_pandas_types=False,
        reset_index=False,
        check_names=False,
    )

    with assert_executed_plan_count(0):
        bres = ba.map_partitions_with_state(
            init_state,
            per_batch,
            1,
            output_type=pd.Series(dtype="string[pyarrow]"),
            bodo=1,
        )

    _test_equal(
        bres,
        res,
        check_pandas_types=False,
        reset_index=False,
        check_names=False,
    )


@pytest.mark.parametrize(
    "quantiles", [[0, 0.25, 0.5, 0.75, 0.9, 1], [0.22, 0.55, 0.99], [0.5]]
)
def test_series_quantile(quantiles):
    """Tests that approximate quantiles using KLL fall within expected error bounds."""

    def kll_error_bounds(q, k=200, pmf=False):
        eps = 1.0 / np.sqrt(k) * (1.7 if pmf else 1.33)
        return max(0.0, q - eps), min(1.0, q + eps)

    df = bd.DataFrame(
        {
            "A": [1] * 30 + [3] * 40 + [5] * 20 + [100] * 10,
            "B": [0.5, 1.5, 2.5] * 30 + [5.5] * 10,
            "C": list(range(100)),
            "D": [100, 200, 300] * 30 + [None] * 10,
            "E": [1] + [100] * 98 + [1000],
        }
    )

    for col in df.columns:
        s = df[col]
        with assert_executed_plan_count(1):
            approx_quantiles = s.quantile(quantiles)

        filtered_list = list(filter(lambda x: x is not pd.NA, s.values))
        sorted_list = sorted(filtered_list)
        n = len(sorted_list)

        assert isinstance(approx_quantiles, bd.Series) and len(approx_quantiles) == len(
            quantiles
        )

        for q in approx_quantiles.index:
            approx = approx_quantiles[q]
            q = float(q)
            lo, hi = kll_error_bounds(q, k=200, pmf=False)

            lo_idx = int(np.floor(lo * (n - 1)))
            hi_idx = int(np.ceil(hi * (n - 1)))

            true_low = sorted_list[lo_idx]
            true_high = sorted_list[hi_idx]

            assert true_low <= approx <= true_high, (
                f"Quantile {q} estimate {approx} not within [{true_low}, {true_high}]"
            )


@pytest.mark.parametrize("q", [0.25, 0.5, 0.75, 0.9])
def test_series_quantile_scalar(q):
    """Tests that approximate quantiles with scalar arguments fall within expected error bounds."""

    def kll_error_bounds(q, k=200, pmf=False):
        eps = 1.0 / np.sqrt(k) * (1.7 if pmf else 1.33)
        return max(0.0, q - eps), min(1.0, q + eps)

    df = bd.DataFrame(
        {
            "A": [1] * 30 + [3] * 40 + [5] * 20 + [100] * 10,
            "B": [0.5, 1.5, 2.5] * 30 + [5.5] * 10,
            "C": list(range(100)),
            "D": [100, 200, 300] * 30 + [None] * 10,
        }
    )

    for col in df.columns:
        s = df[col]
        with assert_executed_plan_count(1):
            approx = s.quantile(q)

        filtered_list = list(filter(lambda x: x is not pd.NA, s.values))
        sorted_list = sorted(filtered_list)
        n = len(sorted_list)

        assert isinstance(approx, float)

        lo, hi = kll_error_bounds(q, k=200, pmf=False)

        lo_idx = int(np.floor(lo * (n - 1)))
        hi_idx = int(np.ceil(hi * (n - 1)))

        true_low = sorted_list[lo_idx]
        true_high = sorted_list[hi_idx]

        assert true_low <= approx <= true_high, (
            f"Quantile {q} estimate {approx} not within [{true_low}, {true_high}]"
        )


def test_series_quantile_empty():
    """Tests that quantile on an empty BodoSeries returns either a scalar pd.NA or a BodoSeries of pd.NA."""

    pds = pd.Series([])
    bds = bd.Series([])

    with assert_executed_plan_count(0):
        pd_quantile = pds.quantile([0.5])
        bodo_quantile = bds.quantile([0.5])

    _test_equal(
        bodo_quantile,
        pd_quantile.astype(pd.ArrowDtype(pa.float64())),
        check_pandas_types=False,
        reset_index=True,
    )

    with assert_executed_plan_count(0):
        pd_quantile = pds.quantile(0.5)
        bodo_quantile = bds.quantile(0.5)

    assert np.isnan(pd_quantile) and bodo_quantile is pd.NA

    pds = pd.Series([1, 2, 3])
    pds = pds[pds > 4]
    bds = bd.Series([1, 2, 3])
    bds = bds[bds > 4]

    with assert_executed_plan_count(1):
        pd_quantile = pds.quantile([0, 0.2, 0.5, 0.8, 1])
        bodo_quantile = bds.quantile([0, 0.2, 0.5, 0.8, 1])

    _test_equal(
        bodo_quantile,
        pd_quantile.astype(pd.ArrowDtype(pa.float64())),
        check_pandas_types=False,
        reset_index=True,
        check_names=False,
    )


def test_series_quantile_tails():
    """Tests that querying quantiles at tail ends return exact values."""

    df = pd.DataFrame({"A": [1] + [100] * 98 + [1000], "B": list(range(100))})
    bdf = bd.from_pandas(df)

    with assert_executed_plan_count(1):
        out_bd = bdf["A"].quantile([0, 1])
        out_pd = df["A"].quantile([0, 1])

    _test_equal(
        out_bd,
        out_pd.astype(pd.ArrowDtype(pa.float64())),
        check_pandas_types=False,
        reset_index=True,
    )


def test_series_quantile_singleton():
    """Tests quantile on a singleton BodoSeries."""

    pds = pd.Series([100])
    bds = bd.Series([100])

    with assert_executed_plan_count(1):
        out_bd = bds.quantile([0, 0.2, 0.6, 0.89, 1])
        out_pd = pds.quantile([0, 0.2, 0.6, 0.89, 1])

    _test_equal(
        out_bd,
        out_pd.astype(pd.ArrowDtype(pa.float64())),
        check_pandas_types=False,
        reset_index=True,
        check_names=False,
    )


@pytest.mark.jit_dependency
def test_dataframe_jit_fallback(datapath):
    """Test fallback to JIT."""
    path = datapath("dataframe_library/df1.parquet")

    bodo_out = bd.read_parquet(path)[["A", "D"]]
    py_out = pd.read_parquet(path, dtype_backend="pyarrow")[["A", "D"]]

    start_jit_fallback_compile_success = JITFallback.compile_success
    with assert_executed_plan_count(1):
        py_dup = py_out.duplicated()
        bd_dup = bodo_out.duplicated()
    end_jit_fallback_compile_success = JITFallback.compile_success

    assert end_jit_fallback_compile_success - start_jit_fallback_compile_success == 1

    _test_equal(
        bd_dup,
        py_dup,
        check_pandas_types=False,
        reset_index=True,
        check_names=False,
    )


@pytest.mark.jit_dependency
def test_top_level_jit_fallback(datapath):
    """Test fallback to JIT."""
    df = pd.DataFrame(
        {
            "A": ["X", "X", "X", "X", "Y", "Y"],
            "B": [1, 2, 3, 4, 5, 6],
            "C": [10, 11, 12, 20, 21, 22],
        }
    )
    bodo_df = bd.DataFrame(df)

    start_jit_fallback_compile_success = JITFallback.compile_success
    with assert_executed_plan_count(1):
        py_pivoted = pd.pivot(df, columns="A", index="B", values="C")
        bodo_pivoted = bd.pivot(bodo_df, columns="A", index="B", values="C")
    end_jit_fallback_compile_success = JITFallback.compile_success

    assert end_jit_fallback_compile_success - start_jit_fallback_compile_success == 1

    bodo_reordered = bodo_pivoted[py_pivoted.columns]
    _test_equal(
        bodo_reordered,
        py_pivoted,
        check_pandas_types=False,
        reset_index=True,
        check_names=False,
        sort_output=True,
    )


@pytest.mark.jit_dependency
def test_set_column_names():
    """Check that setting columns attribute of a DataFrame works as expected."""

    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5] * 2,
            "B": ["A", "B", "C", "D", "E"] * 2,
        },
        index=["aa", "bb", "cc", "dd", "ee"] * 2,
    )

    # Set columns before executing plan
    bdf = bd.from_pandas(df)
    pdf = df.copy()
    new_cols = ["C", "D"]

    with assert_executed_plan_count(0):
        bdf.columns = new_cols
    pdf.columns = new_cols
    _test_equal(bdf.head(0), pdf.head(0), check_pandas_types=False)
    _test_equal(bdf, pdf, check_pandas_types=False)

    # Rename columns while building plan
    bdf = bd.from_pandas(df)
    pdf = df.copy()

    def impl(df):
        df.columns = new_cols
        df2 = df[new_cols]
        return df2

    with assert_executed_plan_count(0):
        bodo_result = impl(bdf)
    pd_result = impl(pdf)
    _test_equal(bodo_result.head(0), pd_result.head(0), check_pandas_types=False)
    _test_equal(bodo_result, pd_result, check_pandas_types=False)

    # Renaming the result of JIT
    @bodo.jit(spawn=True)
    def get_bodo_df(df):
        return df

    bdf = get_bodo_df(df)
    pdf = df.copy()

    with assert_executed_plan_count(0):
        bdf.columns = new_cols
    pdf.columns = new_cols

    _test_equal(bdf.head(0), pdf.head(0), check_pandas_types=False)
    _test_equal(bdf, pdf, check_pandas_types=False)


def test_print_no_warn():
    """Make sure printing a BodoDataFrame or BodoSeries doesn't throw irrelevant
    fallback warnings
    """
    df = bd.DataFrame({"A": np.arange(100) % 30, "B": np.arange(100)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        print(df)

    S = bd.Series(np.arange(100) % 30)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        print(S)


@pytest.mark.parametrize(
    "pd_in, expr",
    [
        pytest.param(
            pd.DataFrame({"A": [1, 2, 3, 4, 5, 6], "B": [1, 2, 3, 10, 20, 30]}),
            lambda x: x[x.B < len(x)],
            id="frame_expr",
        ),
        pytest.param(
            pd.Series([8, 7, 5, 4, 5, 6], name="A"),
            lambda x: x[x < len(x)],
            id="series_expr",
        ),
    ],
)
def test_lazy_len(pd_in, expr, index_val):
    """Make sure len() on BodoDataFrame, BodoSeries doesn't trigger
    execution when used in another plan.
    """
    reset_index = isinstance(index_val, pd.RangeIndex)
    pd_in.index = index_val[: len(pd_in)]

    if isinstance(pd_in, pd.Series):
        pd_in_ser = pd_in.to_frame()
        bd_in = bd.from_pandas(pd_in_ser)
        bd_in = bd_in.A
    else:
        bd_in = bd.from_pandas(pd_in)

    with assert_executed_plan_count(0):
        bodo_out = expr(bd_in)

    pd_out = expr(pd_in)

    _test_equal(bodo_out, pd_out, check_pandas_types=False, reset_index=reset_index)


def test_len_no_warn(index_val):
    """Test that collecting length does not raise warnings"""
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6], "B": [1, 2, 3, 10, 20, 30]})
    df.index = index_val[: len(df)]

    bdf = bd.from_pandas(df)

    with warnings.catch_warnings():
        warnings.simplefilter("error", BodoLibFallbackWarning)
        assert len(bdf[bdf.A > 5]) == len(df[df.A > 5])


@pytest.mark.jit_dependency
def test_bodo_pandas_inside_jit():
    """Make sure using bodo.pandas functions inside a bodo.jit function works as
    expected and is same as pandas.
    """
    df = bd.DataFrame({"A": np.arange(100)})

    def test1(df):
        df2 = bd.DataFrame({"B": np.arange(len(df))})
        return df2.B.sum()

    assert test1(df) == bodo.jit(spawn=False, distributed=False)(test1)(df)

    def test2(df):
        return bd.Timestamp(df.A.iloc[0])

    assert test2(df) == bodo.jit(spawn=False, distributed=False)(test2)(df)


def test_join_non_equi_key_not_in_output():
    """Test for joins with non-equi keys that are not in the output and require special
    handling in column reordering of physical join.
    """

    df1 = pd.DataFrame(
        {
            "A": pd.array([0, 1], "Float64"),
            "B": pd.array([2, 3], "Float64"),
            "C": pd.array([5, 6], "Int32"),
        }
    )
    df2 = pd.DataFrame(
        {
            "D": pd.array([0, 1, 5, 6], "Int32"),
            "E": pd.array([2, 3, 4, 5], "Float64"),
        }
    )

    df3 = df1.merge(df2, left_on="C", right_on="D", how="inner")
    df3 = df3[df3.A < df3.E]

    bdf1 = bd.from_pandas(df1)
    bdf2 = bd.from_pandas(df2)
    bdf3 = bdf1.merge(bdf2, left_on="C", right_on="D", how="inner")
    bdf3 = bdf3[bdf3.A < bdf3.E]

    _test_equal(
        bdf3["B"],
        df3["B"],
        check_pandas_types=False,
        reset_index=True,
        sort_output=True,
    )


def test_series_to_list():
    s1 = pd.Series(list(range(37)))
    bs1 = bd.Series(s1)
    s2 = s1 + 1
    bs2 = bs1 + 1
    l1 = s2.to_list()
    bl1 = bs2.to_list()
    assert l1 == bl1


def test_series_str_match():
    s = pd.Series(["abc", "a1c", "zzz", None], dtype="string")
    bs = bd.Series(s)

    # Match strings that start with 'a' and end with 'c'
    pmask = s.str.match(r"a.*c")
    bmask = bs.str.match(r"a.*c")

    _test_equal(
        bmask,
        pmask,
        check_pandas_types=False,
    )


def test_asarray_df():
    """
    Tests that np.asarray(df) produces the correct ndarray
    when all columns have the same dtype.
    """

    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 4], dtype="Int32"),
            "B": pd.array([5, 6, 7, 8], dtype="Int32"),
            "C": pd.array([9, 10, 11, 12], dtype="Int32"),
        }
    )

    bdf = bd.from_pandas(df)
    result_bodo = np.asarray(bdf)
    result_pandas = np.asarray(df)

    assert result_bodo.shape == (4, 3)
    assert result_bodo.dtype == np.int32
    np.testing.assert_array_equal(result_bodo, result_pandas)


@pytest.fixture
def timezone_timestamp_df():
    """Fixture for DataFrame with timezone-aware timestamps."""
    data = {
        "A": pd.date_range(
            "2023-01-01 00:00:00",
            periods=5,
            freq="h",
            tz="America/New_York",
        ),
        "B": pd.date_range(
            "2023-06-01 12:00:00",
            periods=5,
            freq="D",
            tz="Europe/London",
        ),
    }
    return pd.DataFrame(data)


@pytest.mark.jit_dependency
@pytest.mark.parametrize("engine", ["python", "bodo"])
def test_timezone_scalar_func(engine, index_val, timezone_timestamp_df):
    """Test scalar funcs preserve timezone."""
    df = timezone_timestamp_df
    df.index = index_val[: len(df)]
    bdf = bd.from_pandas(df)

    num_plans = 1 if engine == "python" else 0
    with assert_executed_plan_count(num_plans):
        bodo_out = bdf.A.map(lambda x: x, engine=engine).dt.hour

    expected = df.A.dt.hour
    _test_equal(bodo_out, expected, check_pandas_types=False)


def test_timezone_filter(index_val, timezone_timestamp_df):
    """Test filter works with timezones"""
    df = timezone_timestamp_df
    df.index = index_val[: len(df)]
    bdf = bd.from_pandas(df)

    val = pd.Timestamp("2023-01-01 00:02:00", tz="America/Chicago")
    with assert_executed_plan_count(0):
        bodo_out = bdf[bdf.A > val]
    pandas_out = df[df.A > val]

    reset_index = isinstance(index_val, pd.RangeIndex)
    _test_equal(bodo_out, pandas_out, check_pandas_types=False, reset_index=reset_index)


def test_timezone_groupby(timezone_timestamp_df):
    """Test groupby works with timezones"""
    df = timezone_timestamp_df
    bdf = bd.from_pandas(df)

    with assert_executed_plan_count(0):
        # TODO [BSE-5186]: fix as_index case
        bodo_out = bdf.groupby("A", as_index=False).min().B.dt.hour
    pandas_out = df.groupby("A", as_index=False).min().B.dt.hour

    _test_equal(bodo_out, pandas_out, check_pandas_types=False)


def test_timezone_merge(timezone_timestamp_df):
    """Test merge works with timezone keys"""
    df = timezone_timestamp_df
    df2 = pd.DataFrame(
        {
            "C": pd.date_range(
                "2023-01-01 00:00:00",
                periods=5,
                freq="h",
                tz="America/Chicago",
            )
        }
    )

    bdf = bd.from_pandas(df)
    bdf2 = bd.from_pandas(df2)

    with assert_executed_plan_count(0):
        bodo_out = bdf.merge(bdf2, left_on="A", right_on="C", how="inner")
    pandas_out = df.merge(df2, left_on="A", right_on="C", how="inner")

    _test_equal(
        bodo_out,
        pandas_out,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_empty_df(datapath, index_val):
    """Make sure creating an empty dataframe works"""
    _test_equal(bd.DataFrame(), pd.DataFrame())


def test_np_ufunc(index_val):
    """Test for np.ufunc(Series) case."""

    with assert_executed_plan_count(0):
        df = pd.DataFrame(
            {
                "A": pd.array([0, 1] * 50, "Int32"),
                "B": pd.array([2, 3, 4, 5] * 25, "Float64"),
            }
        )
        df.index = index_val[:100]
        bdf = bd.from_pandas(df)
        bdf["C"] = np.log(bdf["B"] + 1e-8)
        pdf = df.copy()
        pdf["C"] = np.log(pdf["B"] + 1e-8)

    _test_equal(
        bdf,
        pdf,
        check_pandas_types=False,
    )


def test_empty_result(datapath):
    """Test empty result node generated by duckdb."""

    with assert_executed_plan_count(0):
        df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )
        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        pdf = df1[(df1.D > 3) & (df1.D < 2)]
        bdf = bodo_df1[(bodo_df1.D > 3) & (bodo_df1.D < 2)]

    _test_equal(
        bdf,
        pdf,
        check_pandas_types=True,
    )


def test_crossproduct_column_removal(datapath):
    """Test that extra column introduced by crossproduct is removed."""

    with assert_executed_plan_count(0):
        df1 = pd.read_parquet(
            datapath("dataframe_library/df1.parquet"), dtype_backend="pyarrow"
        )
        t = (df1["A"] * df1["D"]).sum() * 2
        df1["prod"] = df1["A"] * df1["D"]
        df2 = df1.groupby("B", as_index=False)["prod"].sum()
        pdf = df2[df2["prod"] == t]

        bodo_df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
        (bodo_df1["A"] * bodo_df1["D"]).sum() * 2
        bodo_df1["prod"] = bodo_df1["A"] * bodo_df1["D"]
        bodo_df2 = bodo_df1.groupby("B", as_index=False)["prod"].sum()
        bdf = bodo_df2[bodo_df2["prod"] == t]

    _test_equal(
        bdf,
        pdf,
        check_pandas_types=True,
    )


def test_join_filter_push_cross_product():
    """Test for join filter pushdown through cross product."""
    df1 = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3], "Int32"),
            "B": pd.array([4, 5, 6], "Int32"),
        }
    )
    df2 = pd.DataFrame(
        {
            "C": pd.array([7, 8], "Int32"),
            "D": pd.array([9, 10], "Int32"),
        }
    )
    df3 = pd.DataFrame(
        {
            "E": pd.array(
                [
                    1,
                    1,
                ],
                "Int32",
            ),
        }
    )
    df4 = df1.merge(df2, how="cross")
    df5 = df4.merge(df3, left_on="A", right_on="E", how="inner")

    with assert_executed_plan_count(0):
        bdf1 = bd.from_pandas(df1)
        bdf2 = bd.from_pandas(df2)
        bdf3 = bd.from_pandas(df3)
        bdf4 = bdf1.merge(bdf2, how="cross")
        bdf5 = bdf4.merge(bdf3, left_on="A", right_on="E", how="inner")

    _test_equal(
        bdf5,
        df5,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_join_filter_pushdown_union():
    """Test for join filter pushdown through union."""
    df1 = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3], "Int32"),
            "B": pd.array([4, 5, 6], "Int32"),
        }
    )
    df2 = pd.DataFrame(
        {
            "A": pd.array([7, 8], "Int32"),
            "B": pd.array([9, 10], "Int32"),
        }
    )
    df3 = pd.DataFrame(
        {
            "C": pd.array(
                [
                    1,
                    7,
                ],
                "Int32",
            ),
        }
    )
    df4 = pd.concat([df1, df2])
    df5 = df4.merge(df3, left_on="A", right_on="C", how="inner")

    with assert_executed_plan_count(0):
        bdf1 = bd.from_pandas(df1)
        bdf2 = bd.from_pandas(df2)
        bdf3 = bd.from_pandas(df3)
        bdf4 = bd.concat([bdf1, bdf2])
        bdf5 = bdf4.merge(bdf3, left_on="A", right_on="C", how="inner")

    _test_equal(
        bdf5,
        df5,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )


def test_join_filter_pushdown_aggregate_split_keys():
    """Test that a join filter that is partially pushed through an aggregate also generates a filter above it with all keys."""
    df1 = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 1], "Int64"),
            "B": pd.array([4, 5, 6, 4], "Int64"),
        }
    )
    df2 = pd.DataFrame(
        {
            "C": pd.array([1, 2], "Int64"),
            "D": pd.array([7, 8], "Int64"),
        }
    )
    df3 = df1.groupby("A", as_index=False).sum()
    df4 = df3.merge(df2, left_on=["A", "B"], right_on=["C", "C"], how="right")

    with assert_executed_plan_count(0):
        bdf1 = bd.from_pandas(df1)
        bdf2 = bd.from_pandas(df2)
        bdf3 = bdf1.groupby("A", as_index=False).sum()
        bdf4 = bdf3.merge(bdf2, left_on=["A", "B"], right_on=["C", "C"], how="right")

    pre, post = bd.plan.getPlanStatistics(bdf4._mgr._plan)
    assert pre == 5
    # One join filter gets inserted above the aggregate to have both keys
    # and one filter gets pushed below the aggregate with only one key
    assert post == 7

    _test_equal(
        bdf4,
        df4,
        check_pandas_types=False,
        sort_output=True,
        reset_index=True,
    )
