import numpy as np
import pandas as pd

from bodo.tests.utils import pytest_slow_unless_window
from bodosql.tests.test_window.window_common import count_window_applies
from bodosql.tests.utils import check_query

# Skip unless any window-related files were changed
pytestmark = pytest_slow_unless_window


def test_lead_lag(spark_info, memory_leak_check):
    """Tests the window functions LEAD/LAG with a minimal set of combinations
    that cover the essential of behavior."""
    selects = [
        "LEAD(I) RESPECT NULLS OVER (PARTITION BY P ORDER BY O)",
        "LAG(S, 10) RESPECT NULLS OVER (PARTITION BY P ORDER BY O)",
        "LEAD(I, 5, 0) RESPECT NULLS OVER (PARTITION BY P ORDER BY O)",
        "LEAD(S, 3) IGNORE NULLS OVER (PARTITION BY P ORDER BY O)",
        "LAG(I, 7, -1) IGNORE NULLS OVER (PARTITION BY P ORDER BY O)",
        "LEAD(S, 1, 'hello') IGNORE NULLS OVER (PARTITION BY P ORDER BY O)",
    ]
    query = f"SELECT I, P, O, {', '.join(selects)} FROM table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "I": pd.Series(
                    [None if (i**2) % 6 < 2 else i for i in range(500)],
                    dtype=pd.Int32Dtype(),
                ),
                "S": pd.Series(
                    [None if (i**3) % 17 > 10 else str(i) for i in range(500)]
                ),
                "P": pd.Series(["A", "B", "C", "D", "E"] * 100),
                "O": pd.Series([np.tan(i) for i in range(500)]),
            }
        )
    }
    pandas_code = check_query(
        query,
        ctx,
        spark_info,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
    )["pandas_code"]

    # Verify that fusion is working correctly so only one closure is produced
    count_window_applies(pandas_code, 1, ["LEAD", "LAG"])
