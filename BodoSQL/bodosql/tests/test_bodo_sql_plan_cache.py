"""
Test that BodoSqlPlanCache works as expected.
"""

import os

import pandas as pd
from mpi4py import MPI

import bodo
import bodosql
from bodo.sql_plan_cache import BodoSqlPlanCache


def test_plan_caching_basic(memory_leak_check, tmp_path):
    """
    Test that the SQL plan is cached at the expected location
    when a query is run as part of a Bodo jit function.
    """
    comm = MPI.COMM_WORLD
    tmp_path_rank0 = comm.bcast(str(tmp_path))

    orig_sql_plan_cache_loc = bodo.sql_plan_cache_loc
    bodo.sql_plan_cache_loc = str(tmp_path_rank0)

    query = "select A, B from table1 where A > 3"
    df = pd.DataFrame(
        {
            "A": pd.Series(range(10), dtype="Int64"),
            "B": pd.Series(["pasta", "pizza"] * 5, dtype="string"),
            "C": pd.Series(range(20, 30), dtype="Int32"),
        }
    )
    bc = bodosql.BodoSQLContext({"TABLE1": df})

    def impl(bc, query):
        return bc.sql(query)

    try:
        # Compile a simple SQL query inside a JIT function
        bodo.jit(impl)(bc, query)
        bodo.barrier()

        # Get the cache location from BodoSqlPlanCache
        expected_cache_loc = BodoSqlPlanCache.get_cache_loc(query)

        # Verify that the file exists. Read the file and do some keyword checks.
        assert os.path.isfile(expected_cache_loc), (
            f"Plan not found at expected cache location ({expected_cache_loc})"
        )

        with open(expected_cache_loc) as f:
            plan_str = f.read()

        bodo.barrier()

        assert "PandasProject(A=[$0], B=[$1])" in plan_str, plan_str
        assert "BodoPhysicalFilter(condition=[>($0, 3)])" in plan_str, plan_str
        # We expect a "verbose" plan with the dtypes and costs to be output
        assert "# types: BIGINT, VARCHAR" in plan_str, plan_str
        assert "# cost:" in plan_str, plan_str

    finally:
        # Restore original value in case of failure
        bodo.sql_plan_cache_loc = orig_sql_plan_cache_loc
