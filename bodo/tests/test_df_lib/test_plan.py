"""
Tests dataframe library plan nodes.
"""

import pyarrow as pa

from bodo.ext import plan_optimizer


def test_join_node():
    """Make sure Cython wrapper around the join node works. Just tests node creation."""
    A = plan_optimizer.LogicalComparisonJoin(plan_optimizer.CJoinType.INNER)
    assert str(A) == "LogicalComparisonJoin(INNER)"


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
