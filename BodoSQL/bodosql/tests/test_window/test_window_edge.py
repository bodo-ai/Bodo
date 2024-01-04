import datetime

import numpy as np
import pandas as pd
import pytest

from bodo.tests.utils import pytest_slow_unless_window
from bodosql.tests.test_window.window_common import (  # noqa
    all_window_df,
    count_window_applies,
    uint8_window_df,
)
from bodosql.tests.utils import check_query

# Skip unless any window-related files were changed
pytestmark = pytest_slow_unless_window


@pytest.mark.skip("TODO: currently defaults to unbounded window in some case")
@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            "SELECT W4, SUM(A) OVER (PARTITION BY W1 ORDER BY W4 ROWS 2 PRECEDING) FROM table1",
            id="sum-only_upper_bound-2_preceding",
        ),
        pytest.param(
            "SELECT W4, AVG(A) OVER (PARTITION BY W1 ORDER BY W4 ROWS 3 FOLLOWING) FROM table1",
            id="avg-only_upper_bound-3_following",
        ),
        pytest.param(
            "SELECT W4, COUNT(A) OVER (PARTITION BY W1 ORDER BY W4 ROWS CURRENT ROW) FROM table1",
            id="count-only_upper_bound-current_row",
        ),
    ],
)
def test_only_upper_bound(query, uint8_window_df, spark_info, memory_leak_check):
    """Tests when only the upper bound is provided to a window function call"""
    check_query(
        query,
        uint8_window_df,
        spark_info,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.skip("TODO")
def test_empty_window(uint8_window_df, spark_info, memory_leak_check):
    """Tests when the clause inside the OVER term is empty"""
    query = "SELECT STDDEV(A) OVER () FROM table1"
    check_query(
        query,
        uint8_window_df,
        spark_info,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_window_no_order(uint8_window_df, spark_info, memory_leak_check):
    """Tests when the window clause does not have an order"""
    query = "SELECT W4, SUM(A) OVER (PARTITION BY W1) FROM table1"
    check_query(
        query,
        uint8_window_df,
        spark_info,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_window_no_rows(uint8_window_df, spark_info, memory_leak_check):
    """Tests when the window clause does not have a rows specification"""
    query = "SELECT W4, SUM(A) OVER (PARTITION BY W1 ORDER BY W4) FROM table1"
    check_query(
        query,
        uint8_window_df,
        spark_info,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.timeout(600)
@pytest.mark.slow
def test_window_case(uint8_window_df, spark_info):
    """Tests windowed window function calls inside of CASE statements. The
       case_args is a list of lists of tuples with the following format:

    [
        [
            ("A", "B"),
            ("C", "D"),
            (None, "E")
        ],
        [
            ("F", "G"),
            (None, "I")
        ]
    ]

    Corresponds to the following query:

    SELECT
        W4,
        CASE
            WHEN A THEN B
            WHEN C THEN D
            ELSE E
        END,
        CASE
            WHEN F THEN G
            ELSE I
        END
    from table1
    """
    cases = []
    window1A = (
        "PARTITION BY W2 ORDER BY W4 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    )
    window1B = "PARTITION BY W2 ORDER BY W4"
    window2 = (
        "PARTITION BY W1 ORDER BY W4 ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING"
    )
    case_args = [
        [
            (f"AVG(A) OVER ({window1A}) > 6.0", "'A+'"),
            (f"AVG(A) OVER ({window1A}) < 4.0", "'A-'"),
            (f"AVG(A) OVER ({window2}) > 6.0", "'B+'"),
            (f"AVG(A) OVER ({window2}) < 4.0", "'B-'"),
            (None, f"'C'"),
        ],
        [
            (f"A < 5", f"COUNT(A) OVER ({window2})"),
            (f"A >= 5", f"COUNT(A) OVER ({window1A})"),
            (None, f"LEAD(A, 3) OVER ({window1B})"),
        ],
    ]
    for case in case_args:
        new_case = ""
        for i, args in enumerate(case):
            if i == len(case) - 1:
                new_case += f"ELSE {args[1]}"
            else:
                new_case += f"WHEN {args[0]} THEN {args[1]} "
        cases.append(f"CASE {new_case} END")
    query = f"SELECT W4, {', '.join(cases)} FROM table1"
    pandas_code = check_query(
        query,
        uint8_window_df,
        spark_info,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        return_codegen=True,
    )["pandas_code"]

    # TODO: enable checking window fusion once window function calls inside
    # of CASE statements can be fused [BE-3962]
    # count_window_applies(pandas_code, 2, ["AVG", "COUNT", "LEAD"])


@pytest.mark.tz_aware
@pytest.mark.slow
def test_tz_aware_partition_by(spark_info):
    """
    Test that tz-aware data can be used as the input to partition by.
    """
    df = pd.DataFrame(
        {
            "TZ": [
                pd.Timestamp("2022-10-1", tz="US/Pacific"),
                pd.Timestamp("2022-1-1", tz="US/Pacific"),
                pd.Timestamp("2022-1-11", tz="US/Pacific"),
                pd.Timestamp("2022-2-1", tz="US/Pacific"),
                pd.Timestamp("2020-1-1", tz="US/Pacific"),
                None,
            ]
            * 5,
            "SORT_COL": np.arange(30),
            "SUM_COL": -np.arange(30, 60),
        }
    )
    ctx = {"TABLE1": df}
    query = "SELECT SORT_COL, SUM(SUM_COL) OVER (PARTITION BY TZ ORDER BY SORT_COL ROWS 2 PRECEDING) FROM TABLE1"
    check_query(
        query,
        ctx,
        spark_info,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        convert_columns_tz_naive=["TZ"],
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "calculation, window",
    [
        pytest.param(
            "AVG(W3 - W4)",
            "PARTITION BY COALESCE(U8, W4 % 10) ORDER BY W4",
            id="partition_function-avg_subtract",
        ),
        pytest.param(
            "SUM(CASE WHEN LEFT(ST, 1) = 'A' THEN 1 ELSE 0 END)",
            "PARTITION BY W2 ORDER BY COALESCE(ST, 'A'), W4",
            id="order_function-sum_case",
        ),
    ],
)
def test_window_using_function(
    calculation, window, all_window_df, spark_info, memory_leak_check
):
    """Tests case where the PARTITION BY or ORDER BY clause uses a function"""
    query = f"SELECT W4, {calculation} OVER ({window} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM table1"
    check_query(
        query,
        all_window_df,
        spark_info,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.parametrize(
    "orders",
    [
        pytest.param(
            [
                "ORDER BY I32",
                "ORDER BY I32 DESC",
                "ORDER BY STR ASC NULLS FIRST, I32 ASC",
                "ORDER BY I32 DESC",
            ],
            id="integers_strings",
        ),
        pytest.param(
            [
                "ORDER BY BIN DESC NULLS FIRST, I32 DESC",
                "ORDER BY DAT ASC NULLS LAST, I32 DESC",
                "ORDER BY BIN DESC NULLS FIRST, I32 DESC",
                "ORDER BY BIN DESC NULLS FIRST, I32 DESC",
            ],
            id="binary_dates",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_row_number_without_partition(orders, spark_info, memory_leak_check):
    """Test using ROW_NUMBER without a partition"""
    query = f"SELECT I32, {', '.join(f'ROW_NUMBER() OVER ({o}) AS C{i}' for i,o in enumerate(orders))} FROM table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "I32": pd.Series(list(range(20)), dtype=pd.Int32Dtype()),
                "STR": (
                    ["A", "e", "I", "o", "U", "y"]
                    + [None] * 4
                    + ["alpha", "A", "gamma", "delta", "epsilon", "U"]
                    + [None] * 4
                ),
                "BIN": [
                    None
                    if i % 4 == 2
                    else bytes(str(2 << int(i / 1.5)), encoding="utf-8")
                    for i in range(20)
                ],
                "DAT": [
                    None
                    if i % 5 == 4
                    else datetime.date.fromordinal(730120 + (7 << (3 * i)) % 10000)
                    for i in range(20)
                ],
            }
        )
    }
    check_query(
        query,
        ctx,
        spark_info,
        sort_output=True,
        check_dtype=False,
        check_names=False,
    )
