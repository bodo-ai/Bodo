"""
Test correctness of Timestamp operations that
include timezone information.

Note: Because Spark does not contain timezones we cannot
compart with SparkSQL for correctness
"""

import pandas as pd
import pytest

from bodo.tests.utils import pytest_slow_unless_codegen
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.mark.slow
def test_select(memory_leak_check):
    """
    Test a simple select statement with a table including
    timezone information.
    """
    query = "Select B from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                "2/2/2022", periods=12, freq="1D2h", tz="Poland", unit="ns"
            ),
            "B": pd.date_range(
                "2/25/2021", periods=12, freq="1D2h", tz="US/Pacific", unit="ns"
            ),
            "C": pd.date_range(
                "5/22/2022", periods=12, freq="1D2h", tz="UTC", unit="ns"
            ),
        }
    )
    expected_output = df[["B"]]
    ctx = {"TABLE1": df}
    check_query(query, ctx, None, expected_output=expected_output)
