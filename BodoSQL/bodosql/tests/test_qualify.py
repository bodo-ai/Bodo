"""
Test correctness of SQL queries containing qualify clauses in BodoSQL
"""

import os

import numpy as np
import pandas as pd
import pytest

from bodosql.tests.utils import check_query, get_equivalent_spark_agg_query

# [BE-3894] TODO: refactor this file like how test_rows.py was refactored
# for window fusion

# Helper environment variable to allow for testing locally, while avoiding
# memory issues on CI
testing_locally = os.environ.get("BODOSQL_TESTING_LOCALLY", False)

from bodo.tests.utils import pytest_slow_unless_window

# Skip unless any window-related files were changed
pytestmark = pytest_slow_unless_window


@pytest.mark.parametrize(
    "func, cmp, use_dummy_frame",
    [
        pytest.param("ROW_NUMBER()", " = 1", False, id="min_row_number_filter"),
        pytest.param("ROW_NUMBER()", " > 1", False, id="other_row_number"),
        pytest.param("CUME_DIST()", " <= 0.5", False, id="cume_dist"),
        pytest.param("NTILE(2)", " = 1", False, id="ntile"),
        pytest.param("MIN(I)", " < 0", True, id="min"),
        pytest.param("MAX(F)", " >= 1", True, id="max"),
        pytest.param("SUM(I)", " >= 0", True, id="sum"),
        pytest.param("STDDEV(F)", " < 3", True, id="stddev"),
        pytest.param("COUNT(S)", " > 2", True, id="count"),
        pytest.param("COUNT(*)", " > 2", True, id="count_star"),
        pytest.param("FIRST_VALUE(I)", " < 0", True, id="first"),
        pytest.param("LAST_VALUE(F)", " < 0", True, id="last"),
        pytest.param("LEAD(I, 1, 0)", " < I", False, id="lead_respect_nulls"),
        pytest.param("LAG(N, 1, 0) IGNORE NULLS", " > 0", False, id="lag_ignore_nulls"),
    ],
)
def test_qualify_no_bounds(func, cmp, use_dummy_frame, spark_info, memory_leak_check):
    """
    A test to ensure qualify works for window functions that do not have specified bounds.

    Generates a query using the parametrized arguments as follows:

    func = ROWS_NUMBER()
    cmp = ' = 1'
    use_dummy_frame = False

    query:
        SELECT
            P, O, I
        FROM table1
        QUALIFY ROW_NUMBER() OVER (PARTITION BY P ORDER BY O) = 1

    However, since Spark does not support QUALIFY syntax, we rewrite this into the following query
    for the purposes of generating a refsol:
        SELECT P, O, I
        FROM (
            SELECT
                P, O, I, ROW_NUMBER() OVER (PARTITION BY P ORDER BY O) AS window_val
            FROM table1)
        WHERE window_val = 1

    If the argument 'use_dummy_frame' is set to True, then the OVER clause of the query will
    include the term "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW.
    """
    order_term = "ORDER BY O"
    if use_dummy_frame:
        order_term += " ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"
    bodosql_query = f"SELECT P, O, I from table1 QUALIFY {func} OVER (PARTITION BY P {order_term}) {cmp}"
    spark_subquery = f"SELECT P, O, I, {func} OVER (PARTITION BY P {order_term}) as window_val from table1"
    spark_query = f"SELECT P, O, I from ({spark_subquery}) where window_val {cmp}"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "P": [1, 1, 2, 1, 1, 2, 3, 2, 1, 1, 2, 3, 4, 3, 2, 1],
                "O": [15, 0, 14, 1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7],
                "I": pd.Series(
                    [-i if i % 3 == 2 else i for i in range(16)], dtype=pd.Int32Dtype()
                ),
                "F": [np.tan(i) for i in range(16)],
                "S": [None if i % 5 > 2 else str(2**i) for i in range(16)],
                "N": pd.Series(
                    [None if i % 3 == 1 else i * (-1) ** i for i in range(16)],
                    dtype=pd.Int32Dtype(),
                ),
                "B": [
                    None if i % 7 > 4 else bytes(str((2**i) % 31), encoding="utf-8")
                    for i in range(16)
                ],
            }
        )
    }
    check_query(
        bodosql_query,
        ctx,
        spark_info,
        equivalent_spark_query=get_equivalent_spark_agg_query(spark_query),
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.parametrize(
    "func, cmp, frame",
    [
        pytest.param("MIN(I)", " > 0", "sliding", id="min-int"),
        pytest.param("MAX(F)", " > 0", "suffix", id="max-float"),
        pytest.param("SUM(F)", " < 0", "sliding", id="sum-float"),
        pytest.param("AVG(I)", " > 1", "prefix", id="avg-int"),
        pytest.param("VARIANCE_POP(F)", " < 10", "suffix", id="var_pop-float"),
        pytest.param("COUNT(S)", " > 1", "prefix", id="count-string"),
        pytest.param("COUNT(B)", " % 2 = 0", "suffix", id="count-binary"),
        pytest.param("COUNT(*)", " = 3", "sliding", id="count_star"),
        pytest.param("FIRST_VALUE(S)", " IS NULL", "sliding", id="first-string"),
        pytest.param("LAST_VALUE(S)", " IS NOT NULL", "sliding", id="last-string"),
    ],
)
def test_qualify_with_bounds(func, cmp, frame, spark_info, memory_leak_check):
    """
    A test to ensure qualify works for window functions using window frame bounds.
    Specifically, tests with one of 3 common window frame patterns:
    - prefix: the current row + all rows before it
    - suffix: the current row + all rows after it
    - sliding: a narrow range of rows centered on the current row

    Generates a query using the parametrized arguments as follows:

    func = MIN(I)
    cmp = " > 0"
    frame = "prefix"

    query:
        SELECT
            P, O
        FROM table1
        QUALIFY MIN(I) OVER (PARTITION BY P ORDER BY O ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) > 0

    However, since Spark does not support QUALIFY syntax, we rewrite this into the following query
    for the purposes of generating a refsol:
        SELECT P, O
        FROM (
            SELECT
                P, O, MIN(I) OVER (PARTITION BY P ORDER BY O ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) > 0 AS window_val
            FROM table1)
        WHERE window_val > 0
    """
    order_term = "ORDER BY O"
    if frame == "prefix":
        order_term += " ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    elif frame == "suffix":
        order_term += " ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING"
    elif frame == "sliding":
        order_term += " ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING"
    bodosql_query = f"SELECT P, O from table1 QUALIFY {func} OVER (PARTITION BY P {order_term}) {cmp}"
    spark_subquery = f"SELECT P, O, {func} OVER (PARTITION BY P {order_term}) as window_val from table1"
    spark_query = f"SELECT P, O from ({spark_subquery}) where window_val {cmp}"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "P": [1, 1, 2, 1, 1, 2, 3, 2, 1, 1, 2, 3, 4, 3, 2, 1],
                "O": [15, 0, 14, 1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7],
                "I": pd.Series(
                    [-i if i % 3 == 2 else i for i in range(16)], dtype=pd.Int32Dtype()
                ),
                "F": [np.tan(i) for i in range(16)],
                "S": [None if i % 5 > 2 else str(2**i) for i in range(16)],
                "B": [
                    None if i % 7 > 4 else bytes(str((2**i) % 31), encoding="utf-8")
                    for i in range(16)
                ],
            }
        )
    }
    check_query(
        bodosql_query,
        ctx,
        spark_info,
        equivalent_spark_query=get_equivalent_spark_agg_query(spark_query),
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.skip("TODO(BSE-5301): Fix remaining test issues from Pandas 3.0")
@pytest.mark.slow
def test_qualify_timedelta(
    memory_leak_check,
):
    """
    Tests QUALIFY works on windowed aggregations when both bounds are specified on timedelta types.
    """

    query = "SELECT P, I from table1 QUALIFY MIN(TD) OVER (PARTITION BY P) >= INTERVAL '1' DAYS"
    df = pd.DataFrame(
        {
            "P": [1, 1, 2, 1, 1, 2, 3, 2, 1, 1, 2, 3, 4, 3, 2, 1],
            "I": list(range(16)),
            "TD": [np.timedelta64(2**i, "h") for i in range(16)],
        }
    )
    ctx = {"TABLE1": df}
    answer = df.iloc[[6, 11, 12, 13], [0, 1]]

    check_query(
        query,
        ctx,
        None,
        expected_output=answer,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


"""
    The following tests ensures that QUALIFY is evaluated in the proper ordering.
    From snowflake docs, the evaluation order of a query should be as follows
        From
        Where
        Group by
        Having
        Window
        QUALIFY
        Distinct
        Order by
        Limit

    Note: We can't test order by, as, in bodosql, if we don't provide an ordering in the WINDOW clause
    itself, then the ordering is undefined when evaluating the window function.
"""


def test_QUALIFY_eval_order_WHERE(spark_info, memory_leak_check):
    """Ensures that WHERE is evaluated before QUALIFY"""
    df = pd.DataFrame(
        {
            "A": [1, 2, 3] * 4,
            "B": [1] * 12,
        }
    )

    ctx = {"TABLE1": df}

    # If QUALIFY is evaluated first, MAX(A) OVER (PARTITION BY B) = 3 everywhere,
    # as B is one group. If evaluated after where, MAX(A) OVER (PARTITION BY B)
    # = 2 everywhere
    query = "SELECT A from table1 where A < 3 QUALIFY MAX(A) OVER (PARTITION BY B) = 3"

    expected_output = pd.DataFrame(
        {
            "A": [],
        }
    )

    check_query(
        query,
        ctx,
        spark_info,
        expected_output=expected_output,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


def test_QUALIFY_eval_order_GROUP_BY_HAVING(spark_info, memory_leak_check):
    """Ensures that Group by and HAVING are evaluated before QUALIFY"""
    df = pd.DataFrame(
        {
            "A": [1, 2, 3] * 4,
            "B": [1] * 12,
        }
    )

    ctx = {"TABLE1": df}

    # If QUALIFY is evaluated first, MAX(A) OVER (PARTITION BY B) = 3 everywhere,
    # as B is one group. If evaluated after HAVING/GROUP BY, MAX(A) OVER (PARTITION BY B)
    # = 2 everywhere, as the HAVING clause will have eliminated the A=3 rows

    # (GROUP BY A, B HAVING MAX(A) > 3) is just a fancy way of doing WHERE A > 3 in this case
    query = "SELECT A from table1 GROUP BY A, B HAVING MAX(A) > 3 QUALIFY MAX(A) OVER (PARTITION BY B) = 3"

    expected_output = pd.DataFrame(
        {
            "A": [],
        }
    )

    check_query(
        query,
        ctx,
        spark_info,
        expected_output=expected_output,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_QUALIFY_eval_order_DISTINCT(spark_info, memory_leak_check):
    """Ensures that DISTINCT is evaluated after QUALIFY"""
    df = pd.DataFrame(
        {
            "A": [1] * 4 + [2] * 4 + [3] * 4,
            "B": [1] * 12,
        }
    )

    ctx = {"TABLE1": df}

    """
    If distinct is evaluated before QUALIFY, the dataframe will look something like:
    pd.DataFrame({
        "A" : [1,2,3]
    })
    And then the qualify filtering will reduce it to an empty dataframe.

    If qualify is evaluated first, then the dataframe will be unchanged,
    and the distinct clause will result in an output of
    pd.DataFrame({
        "A" : [1,2,3]
    })

    """

    query = "SELECT DISTINCT A from table1 QUALIFY COUNT(A) OVER (PARTITION BY B) > 3"

    expected_output = pd.DataFrame({"A": [1, 2, 3]})

    check_query(
        query,
        ctx,
        spark_info,
        expected_output=expected_output,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


def test_QUALIFY_eval_order_LIMIT(spark_info, memory_leak_check):
    """Ensures that LIMIT is evaluated after QUALIFY"""
    df = pd.DataFrame(
        {
            "A": range(18),
            "B": [1] * 4 + [2] * 3 + [3] * 5 + [4] * 6,
        }
    )

    ctx = {"TABLE1": df}

    """If limit is evaluated first, we expect:
     df = pd.DataFrame({
        "A" : [0, 1, 2],
        "B" : [1] * 3,
    })
    and the qualify Count filtering will return those rows from column A.

    If qualify is evaluated first, we expect the output after evaluating the
    qualify to be as follows, and then the limit will output the same rows:
    df = pd.DataFrame({
        "A" : [4, 5, 6],
        "B" : [2] * 3,
    })

    """
    query = "SELECT A from table1 QUALIFY COUNT(A) OVER (PARTITION BY B) <= 3"

    expected_output = pd.DataFrame({"A": [4, 5, 6]})

    check_query(
        query,
        ctx,
        spark_info,
        expected_output=expected_output,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_QUALIFY_nested_queries(spark_info, memory_leak_check):
    """stress test to ensure that qualify works with nested subqueries"""

    table1 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7] * 3,
            "B": [1, 1, 2, 2, 3, 3, 4] * 3,
            "C": [1, 1, 1, 2, 2, 3, 3] * 3,
        }
    )

    ctx = {"TABLE1": table1}

    bodosql_query1 = "SELECT ROW_NUMBER() OVER (PARTITION BY B ORDER BY C, B, A) as w FROM table1 QUALIFY w < 10"
    bodosql_query2 = f"SELECT MAX(A) over (PARTITION BY C ORDER BY B, C, A) as x FROM table1 QUALIFY x in ({bodosql_query1})"
    bodosql_query = bodosql_query2

    spark_query_1 = "SELECT * FROM (SELECT ROW_NUMBER() OVER (PARTITION BY B ORDER BY C, B, A) as w FROM table1) WHERE w < 10"
    spark_query_2 = f"SELECT * FROM (SELECT MAX(A) over (PARTITION BY C ORDER BY B, C, A) as x FROM table1 ) WHERE x in ({spark_query_1})"
    spark_query = spark_query_2

    check_query(
        bodosql_query,
        ctx,
        spark_info,
        equivalent_spark_query=spark_query,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.tz_aware
def test_qualify_tz_aware(memory_leak_check):
    """Tests that qualify is supported with tz-aware data."""
    query = "SELECT A, MIN(A) over (PARTITION BY C ORDER BY B ASC ROWS BETWEEN 1 PRECEDING and 1 FOLLOWING) as x FROM table1 QUALIFY x > A"
    tz = "US/Pacific"
    df = pd.DataFrame(
        {
            "A": [
                pd.Timestamp(year=2022, month=1, day=1, tz=tz),
                pd.Timestamp(year=2022, month=1, day=2, tz=tz),
                pd.Timestamp(year=2022, month=11, day=1, tz=tz),
                None,
                pd.Timestamp(year=2022, month=1, day=15, tz=tz),
            ]
            * 4,
            "B": np.arange(20),
            "C": ["left", "right", "left", "left"] * 5,
        }
    )
    ctx = {"TABLE1": df}
    # Compute the expected output of the max. To do this we leverage
    # that the window is the current previous and next value and the
    # Order by keeps the DataFrame in order.
    x_list = []
    right_offset = 4
    right_modulo = 1
    for i in range(len(df)):
        # Determine if we are grouped by left or right
        group = df["C"].iat[i]
        if group == "right":
            prev = i - right_offset
            next = i + right_offset
        else:
            prev = i - 1
            next = i + 1
            if prev % right_offset == right_modulo:
                prev -= 1
            if next % right_offset == right_modulo:
                next += 1
        options = []
        if prev > 0:
            options.append(df["A"].iat[prev])
        options.append(df["A"].iat[i])
        if next < len(df):
            options.append(df["A"].iat[next])
        # Remove NA values as options
        options = [x for x in options if pd.notna(x)]
        x_list.append(min(options))
    py_output = pd.DataFrame(
        {
            "A": df["A"],
            "X": x_list,
        }
    )
    # Now apply the qualify filter
    filter = py_output["X"] > py_output["A"]
    py_output = py_output[filter]
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
        only_jit_1DVar=True,
    )
