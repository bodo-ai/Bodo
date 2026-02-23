"""
Test example functions that mix SQL and Python inside
JIT functions.
"""

import pandas as pd

import bodosql
from bodo.tests.utils import check_func, pytest_mark_one_rank
from bodosql.imported_java_classes import JavaEntryPoint


def test_count_head(datapath, memory_leak_check):
    """
    Check that computing an aggregation is compatible with
    a function like head that modify the index.

    This bug was spotted by Anudeep while trying to breakdown
    a query into components.

    """

    def impl(filename):
        bc = bodosql.BodoSQLContext({"T1": bodosql.TablePath(filename, "parquet")})
        df = bc.sql("select count(B) as cnt from t1")
        return df.head()

    filename = datapath("sample-parquet-data/no_index.pq")
    read_df = pd.read_parquet(filename, dtype_backend="pyarrow")
    count = read_df.B.count()
    expected_output = pd.DataFrame({"CNT": count}, index=pd.Index([0]))
    check_func(impl, (filename,), py_output=expected_output, is_out_distributed=False)


@pytest_mark_one_rank
def test_planner_reset(memory_leak_check):
    """Tests that we have an API exposed that allows resetting the planner,
    allowing parsing multiple queries without crashing."""
    generator = bodosql.BodoSQLContext()._create_generator(False)
    queries = ["SELECT 1", "SELECT COUNT(*) FROM TABLE1", "Select MAX(A) FROM TABLE1"]
    final_query = "DESCRIBE TABLE MYTABLE"
    for query in queries:
        JavaEntryPoint.parseQuery(generator, query)
        JavaEntryPoint.resetPlanner(generator)
    JavaEntryPoint.parseQuery(generator, final_query)
    # Verify that the internal state is based on the final query
    # (e.g. its been reset). All other queries are not DDL.
    assert JavaEntryPoint.isDDLProcessedQuery(generator)
