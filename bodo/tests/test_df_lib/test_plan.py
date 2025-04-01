"""
Tests dataframe library plan nodes.
"""

import operator

import pyarrow as pa

from bodo.ext import plan_optimizer


def test_join_node():
    """Make sure Cython wrapper around the join node works. Just tests node creation."""
    P1 = plan_optimizer.LogicalGetParquetRead(b"example.parquet1")
    P2 = plan_optimizer.LogicalGetParquetRead(b"example.parquet2")
    A = plan_optimizer.LogicalComparisonJoin(
        P1, P2, plan_optimizer.CJoinType.INNER, [(0, 0)]
    )
    assert str(A) == "LogicalComparisonJoin(INNER)"


def test_projection_node():
    """Make sure Cython wrapper around the projection node works. Just tests node creation."""
    P1 = plan_optimizer.LogicalGetParquetRead(b"example.parquet1")
    A = plan_optimizer.LogicalProjection(P1, [(1, "int64"), (3, "string")])
    assert (
        str(A)
        == "LogicalProjection([1, 3], [<CLogicalTypeId.BIGINT: 14>, <CLogicalTypeId.VARCHAR: 25>])"
    )


def test_filter_node():
    """Make sure Cython wrapper around the filter node works. Just tests node creation."""
    P1 = plan_optimizer.LogicalGetParquetRead(b"example.parquet1")
    A = plan_optimizer.LogicalProjection(P1, [(0, "int64")])
    B = plan_optimizer.LogicalBinaryOp(A, 5, operator.gt)
    C = plan_optimizer.LogicalFilter(P1, B)
    assert str(C) == "LogicalFilter()"


def test_parquet_node():
    """Make sure Cython wrapper around the Parquet node works. Just tests node creation."""
    A = plan_optimizer.LogicalGetParquetRead(
        b"example.parquet", pa.schema([("A", pa.int64()), ("B", pa.string())])
    )
    assert str(A) == "LogicalGetParquetRead(example.parquet)"
    assert A.path == "example.parquet"


def test_optimize_call():
    """Make sure Cython wrapper around optimize call works."""
    A = plan_optimizer.LogicalGetParquetRead(
        b"example.parquet", pa.schema([("A", pa.int64()), ("B", pa.string())])
    )
    B = plan_optimizer.py_optimize_plan(A)
    assert str(B) == "LogicalOperator()"
