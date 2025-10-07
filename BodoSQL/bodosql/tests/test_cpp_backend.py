"""
Test BodoSQL C++ backend
"""

import pandas as pd
import pytest

from bodosql.tests.utils import check_query

pytestmark = pytest.mark.bodosql_cpp


def test_basic_query(memory_leak_check):
    """
    Simple test to ensure C++ backend is working
    """
    ctx = {"TABLE1": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})}
    query = "SELECT A+B FROM TABLE1"
    out = pd.DataFrame({"output": [5, 7, 9]})
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        only_python=True,
        expected_output=out,
    )
