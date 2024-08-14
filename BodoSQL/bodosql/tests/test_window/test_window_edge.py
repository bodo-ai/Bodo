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


def test_window_no_rows(uint8_window_df, spark_info, memory_leak_check):
    """Tests when the window clause does not have order/rows specification"""
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


def test_window_pruning_multiple_layers(spark_info, memory_leak_check):
    """
    Tests a query that will result in a plan with multiple layers
    of projection, filter & window nodes, with complex rel trimming
    behavior.
    """
    query = """
    SELECT A, E, P3, O4, W1, W5, W8
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY O4 ORDER BY O5) AS RN
        FROM (
            SELECT 
                A, B, E,
                P1, P3, O4, P5, O5,
                W1,
                LEAD(W1 - A) OVER (PARTITION BY P1 ORDER BY O2) as W6,
                W3,
                LEAD(W3 + C, 1, -1) OVER (PARTITION BY P3 ORDER BY O2) as W7,
                W4,
                LEAD(W2 + W4, -1, -1) OVER (PARTITION BY P2 ORDER BY O2) as W8,
                W5
            FROM (
                SELECT 
                    P1, P2, P3, P4, P5,
                    O1, O2, O3, O4, O5,
                    LEAD(A) OVER (PARTITION BY P1 ORDER BY O1+O2) as W1,
                    A,
                    LEAD(B, 2) OVER (PARTITION BY P2 ORDER BY O1+O2) as W2,
                    B,
                    LEAD(C) OVER (PARTITION BY P3-P1 ORDER BY O3-O5) as W3,
                    C,
                    LEAD(D) OVER (PARTITION BY P4 ORDER BY O4) as W4,
                    D,
                    LEAD(E) OVER (PARTITION BY P5 ORDER BY O5) as W5,
                    E
                FROM TABLE1
                WHERE O1 >= O2
            )
            QUALIFY RANK() OVER (PARTITION BY P1 ORDER BY O2) = DENSE_RANK() OVER (PARTITION BY P1 ORDER BY O2)
        )
    )
    WHERE RN = 1
    """
    spark_query = """
    SELECT A, E, P3, O4, W1, W5, W8
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY O4 ORDER BY O5) AS RN
        FROM (
            SELECT 
                A, B, E,
                P1, P3, O4, P5, O5,
                W1,
                LEAD(W1 - A) OVER (PARTITION BY P1 ORDER BY O2) as W6,
                W3,
                LEAD(W3 + C, 1, -1) OVER (PARTITION BY P3 ORDER BY O2) as W7,
                W4,
                LEAD(W2 + W4, -1, -1) OVER (PARTITION BY P2 ORDER BY O2) as W8,
                W5
            FROM (
                SELECT 
                    P1, P2, P3, P4, P5,
                    O1, O2, O3, O4, O5,
                    LEAD(A) OVER (PARTITION BY P1 ORDER BY O1+O2) as W1,
                    A,
                    LEAD(B, 2) OVER (PARTITION BY P2 ORDER BY O1+O2) as W2,
                    B,
                    LEAD(C) OVER (PARTITION BY P3-P1 ORDER BY O3-O5) as W3,
                    C,
                    LEAD(D) OVER (PARTITION BY P4 ORDER BY O4) as W4,
                    D,
                    LEAD(E) OVER (PARTITION BY P5 ORDER BY O5) as W5,
                    E,
                    RANK() OVER (PARTITION BY P1 ORDER BY O2) = DENSE_RANK() OVER (PARTITION BY P1 ORDER BY O2) as COND
                FROM TABLE1
                WHERE O1 >= O2
            )
            WHERE cond
        )
    )
    WHERE RN = 1
    """
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "P1": [0] * 5,
                "P2": [1] * 5,
                "P3": [2] * 5,
                "P4": [3] * 5,
                "P5": [4] * 5,
                "O1": [0, 1, 2, 3, 4],
                "O2": [0, 1, 2, 3, 4],
                "O3": [0, 1, 2, 3, 4],
                "O4": [0, 1, 2, 3, 4],
                "O5": [0, 1, 2, 3, 4],
                "A": pd.array([2**i for i in range(5)], dtype=pd.Int16Dtype()),
                "B": pd.array([3**i for i in range(5)], dtype=pd.Int16Dtype()),
                "C": pd.array([4**i for i in range(5)], dtype=pd.Int16Dtype()),
                "D": pd.array([5**i for i in range(5)], dtype=pd.Int16Dtype()),
                "E": pd.array([6**i for i in range(5)], dtype=pd.Int16Dtype()),
            }
        )
    }
    check_query(
        query,
        ctx,
        spark_info,
        equivalent_spark_query=spark_query,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


def test_window_pruning_single_layer(spark_info, memory_leak_check):
    """
    Variant of test_window_pruning_multiple_layers but with a simpler query
    with only 1 layer of window calls.
    """
    query = """
    SELECT 
        A,
        B,
        W2,
        P1,
        O2
    FROM (
        SELECT
            A,
            B,
            LEAD(A, 1) OVER (PARTITION BY P1 ORDER BY O1) as W1,
            LAG(B, 1, 0) OVER (PARTITION BY P2 ORDER BY O2) as W2,
            P1,
            P2,
            O1,
            O2
        FROM TABLE1
    )
    """
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "P1": [0] * 5,
                "P2": [1] * 5,
                "O1": [1, 2, 3, 4, 5],
                "O2": [10, 20, 30, 40, 50],
                "A": pd.array([2**i for i in range(5)], dtype=pd.Int16Dtype()),
                "B": pd.array([3**i for i in range(5)], dtype=pd.Int16Dtype()),
            }
        )
    }
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


def test_window_max_over_empty(memory_leak_check):
    """
    Tests calling MAX as a window function without a partition or orderby.
    """
    query = """
    SELECT IDX, MAX(DATA) OVER () as WIN
    FROM TABLE1
    """
    n_rows = 1_000_000
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "IDX": range(n_rows),
                "DATA": [
                    str(i).replace("9", "").replace("6", "").replace("88", "4")
                    for i in range(n_rows)
                ],
            }
        )
    }
    max_val = ctx["TABLE1"]["DATA"].max()
    answer = pd.DataFrame(
        {
            "IDX": range(n_rows),
            "WIN": [max_val for _ in range(n_rows)],
        }
    )
    check_query(
        query,
        ctx,
        None,
        expected_output=answer,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )
