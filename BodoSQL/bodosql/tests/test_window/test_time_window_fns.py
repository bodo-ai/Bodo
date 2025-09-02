import pandas as pd
import pytest

# Skip unless any window-related files were changed
from bodo.tests.utils import pytest_mark_multi_rank_nightly, pytest_slow_unless_window
from bodo.types import Time
from bodosql.tests.test_window.window_common import count_window_applies
from bodosql.tests.utils import check_query

pytestmark = pytest_slow_unless_window


@pytest.fixture
def time_df():
    # To be used for partitioning
    A = pd.Series(
        [
            Time(12, 0, 0),
            Time(12, 0, 0),
            Time(20, 30, 59, microsecond=999),
            Time(20, 30, 59, microsecond=999),
            Time(20, 30, 59, microsecond=999),
            Time(12, 0, 0),
            Time(12, 0, 0),
        ]
    )
    # To be used for ordering or aggregation
    B = pd.Series(
        [
            Time(1, 0, 0),
            Time(3, 10, 4),
            Time(15, 5, 1, nanosecond=1),
            Time(14, 15, 9),
            Time(4, 20, 16, millisecond=250),
            Time(13, 25, 25),
            Time(6, 30, 49),
        ]
    )
    # A numerical column to partition on
    C = pd.Series([1, 2, 1, 2, 1, 2, 1])
    # A numerical column to aggregate or order by
    D = pd.Series([10**i for i in range(7)])
    # To be used for aggregations, including NULLs
    E = pd.Series(
        [
            Time(1, 0, 0),
            None,
            None,
            None,
            Time(5, 0, 0),
            Time(6, 0, 0),
            None,
        ]
    )
    return pd.DataFrame({"A": A, "B": B, "C": C, "D": D, "E": E})


@pytest.mark.slow
def test_time_partition(time_df, memory_leak_check):
    """Tests window functions where the PARTITION BY term is a time column"""
    query = "SELECT A, B, SUM(D) OVER (PARTITION BY A ORDER BY D ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM table1"
    ctx = {"TABLE1": time_df}
    answer = [1, 11, 100, 1100, 11100, 100011, 1100011]
    expected_output = pd.DataFrame({"A": time_df.A, "B": time_df.B, "C": answer})
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
        sort_output=True,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_time_order(time_df, memory_leak_check):
    """Tests window functions where the ORDER BY term is a time column"""
    query = "SELECT C, B, SUM(D) OVER (PARTITION BY C ORDER BY B ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM table1"
    ctx = {"TABLE1": time_df}
    answer = [1, 10, 1010101, 101010, 10001, 100010, 1010001]
    expected_output = pd.DataFrame({"C": time_df.C, "B": time_df.B, "P": answer})
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
@pytest_mark_multi_rank_nightly
def test_time_min_max_count_lead_lag(time_df, memory_leak_check):
    """Tests the window functions MIN, MAX, COUNT, LEAD and LAG with time inputs"""
    query = "SELECT C, D, \
             MIN(B) OVER (PARTITION BY C ORDER BY D ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), \
             MAX(B) OVER (PARTITION BY C ORDER BY D ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), \
             COUNT(E) OVER (PARTITION BY C ORDER BY D ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), \
             LEAD(B) OVER (PARTITION BY C ORDER BY D), \
             LAG(B) OVER (PARTITION BY C ORDER BY D) \
             FROM table1"
    ctx = {"TABLE1": time_df}
    mins = [
        Time(1, 0, 0),
        Time(3, 10, 4),
        Time(1, 0, 0),
        Time(3, 10, 4),
        Time(1, 0, 0),
        Time(3, 10, 4),
        Time(1, 0, 0),
    ]
    maxs = [
        Time(1, 0, 0),
        Time(3, 10, 4),
        Time(15, 5, 1, nanosecond=1),
        Time(14, 15, 9),
        Time(15, 5, 1, nanosecond=1),
        Time(14, 15, 9),
        Time(15, 5, 1, nanosecond=1),
    ]
    counts = [1, 0, 1, 0, 2, 1, 2]
    leads = [
        Time(15, 5, 1, nanosecond=1),
        Time(14, 15, 9),
        Time(4, 20, 16, millisecond=250),
        Time(13, 25, 25),
        Time(6, 30, 49),
        None,
        None,
    ]
    lags = [
        None,
        None,
        Time(1, 0, 0),
        Time(3, 10, 4),
        Time(15, 5, 1, nanosecond=1),
        Time(14, 15, 9),
        Time(4, 20, 16, millisecond=250),
    ]
    expected_output = pd.DataFrame(
        {
            "C": time_df.C,
            "D": time_df.D,
            "MI": mins,
            "MA": maxs,
            "CO": counts,
            "LE": leads,
            "LA": lags,
        }
    )
    pandas_code = check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        return_codegen=True,
    )["pandas_code"]

    # Verify that fusion is working correctly.
    count_window_applies(pandas_code, 1, ["MIN", "MAX", "MODE"])
