"""
Tests dataframe library plan nodes.
"""

import operator

import pyarrow as pa

from bodo.ext import plan_optimizer


def test_join_node(datapath):
    """Make sure Cython wrapper around the join node works. Just tests node creation."""
    P1 = plan_optimizer.LogicalGetParquetRead(
        pa.schema([("A", pa.int64()), ("B", pa.string())]),
        datapath("example.parquet"),
        {},
    )
    P2 = plan_optimizer.LogicalGetParquetRead(
        pa.schema([("A", pa.int64()), ("B", pa.string())]),
        datapath("example2.parquet"),
        {},
    )
    A = plan_optimizer.LogicalComparisonJoin(
        pa.schema([("A", pa.int64()), ("B", pa.string())]),
        P1,
        P2,
        plan_optimizer.CJoinType.INNER,
        [(0, 0)],
    )
    assert str(A) == "LogicalComparisonJoin(INNER)"


def test_projection_node(datapath):
    """Make sure Cython wrapper around the projection node works. Just tests node creation."""
    P1 = plan_optimizer.LogicalGetParquetRead(
        pa.schema([("A", pa.int64()), ("B", pa.string())]),
        datapath("example.parquet"),
        {},
    )
    exprs = [
        plan_optimizer.ColRefExpression(pa.schema([("A", pa.int64())]), P1, 0),
        plan_optimizer.ColRefExpression(pa.schema([("B", pa.string())]), P1, 1),
    ]
    A = plan_optimizer.LogicalProjection(
        pa.schema([("A", pa.int64()), ("C", pa.string())]),
        P1,
        exprs,
    )
    assert str(A) == "LogicalProjection(A: int64\nC: string)"


def test_filter_node(datapath):
    """Make sure Cython wrapper around the filter node works. Just tests node creation."""
    P1 = plan_optimizer.LogicalGetParquetRead(
        pa.schema([("A", pa.int64()), ("B", pa.string())]),
        datapath("example.parquet"),
        {},
    )
    A = plan_optimizer.ColRefExpression(pa.schema([("A", pa.int64())]), P1, 0)
    B = plan_optimizer.ComparisonOpExpression(
        pa.schema([("A", pa.bool_())]), A, 5, operator.gt
    )
    C = plan_optimizer.LogicalFilter(pa.schema([("A", pa.int64())]), P1, B)
    assert str(C) == "LogicalFilter()"


def test_parquet_node(datapath):
    """Make sure Cython wrapper around the Parquet node works. Just tests node creation."""
    A = plan_optimizer.LogicalGetParquetRead(
        pa.schema([("A", pa.int64()), ("B", pa.string())]),
        datapath("example.parquet"),
        {},
    )
    assert str(A).startswith("LogicalGetParquetRead(") and str(A).endswith(
        "example.parquet)"
    )
    assert A.path.endswith("example.parquet")


def test_optimize_call(datapath):
    """Make sure Cython wrapper around optimize call works."""
    A = plan_optimizer.LogicalGetParquetRead(
        pa.schema([("A", pa.int64()), ("B", pa.string())]),
        datapath("example.parquet"),
        {},
    )
    B = plan_optimizer.py_optimize_plan(A)
    assert str(B) == "LogicalOperator()"


def test_parquet_projection_pushdown(datapath):
    """Make sure Projection pushdown works for Parquet read."""
    A = plan_optimizer.LogicalGetParquetRead(
        pa.schema(
            [
                ("A", pa.int64()),
                ("B", pa.string()),
                ("C", pa.int32()),
                ("D", pa.int32()),
            ]
        ),
        datapath("example.parquet"),
        {},
    )
    exprs = [
        plan_optimizer.ColRefExpression(pa.schema([("A", pa.int64())]), A, 0),
        plan_optimizer.ColRefExpression(pa.schema([("C", pa.int32())]), A, 2),
    ]
    B = plan_optimizer.LogicalProjection(
        pa.schema([("A", pa.int64()), ("C", pa.int32())]), A, exprs
    )
    C = plan_optimizer.py_optimize_plan(B)
    assert plan_optimizer.get_pushed_down_columns(C) == [0, 2], (
        "Invalid projection pushdown"
    )
