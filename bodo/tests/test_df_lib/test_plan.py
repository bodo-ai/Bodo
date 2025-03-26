"""
Tests dataframe library plan nodes.
"""

import bodo.pandas.plan_optimizer


def test_join_node():
    """Make Sure Cython wrapper around the join node works. Just tests node creation."""
    A = bodo.pandas.plan_optimizer.LogicalComparisonJoin(
        bodo.pandas.plan_optimizer.CJoinType.INNER
    )
    assert str(A) == "LogicalComparisonJoin(INNER)"


def test_parquet_node():
    """Make Sure Cython wrapper around the Parquet node works. Just tests node creation."""
    A = bodo.pandas.plan_optimizer.LogicalGetParquetRead(b"example.parquet")
    assert str(A) == "LogicalGetParquetRead(example.parquet)"
    assert A.path == "example.parquet"
