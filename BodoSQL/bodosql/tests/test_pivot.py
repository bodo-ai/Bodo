import pandas as pd
import pytest

from bodosql.tests.utils import check_query


def test_basic_pivot(basic_df, spark_info, memory_leak_check):
    """
    Basic test for PIVOT with 1 column.
    """
    query = """
    SELECT * FROM table1
    PIVOT (
        SUM(A) AS SUM_A, AVG(C) AS AVG_C
        FOR A IN (1 as SINGLE, 3 as TRIPLE)
    )
    """
    check_query(query, basic_df, spark_info, check_dtype=False)


def test_multi_col_pivot(basic_df, spark_info, memory_leak_check):
    """
    Basic test for PIVOT with multiple

     columns.
    """
    query = """
    SELECT * FROM table1
    PIVOT (
        SUM(A) AS SUM_A, AVG(C) AS AVG_C
        FOR (A, B) IN ((1, 4) as COL1, (2, 5) as COL2)
    )
    """
    check_query(
        query, basic_df, spark_info, check_dtype=False, is_out_distributed=False
    )


@pytest.mark.slow
def test_basic_null_pivot(basic_df, spark_info, memory_leak_check):
    """
    Basic test for PIVOT with 1 column without a match somewhere.
    """
    query = """
    SELECT * FROM table1
    PIVOT (
        SUM(A) AS SUM_A, AVG(C) AS AVG_C
        FOR A IN (1 as SINGLE, 7 as TRIPLE)
    )
    """
    # set check_dtype=False because of int64 vs Int64 difference
    check_query(query, basic_df, spark_info, check_dtype=False)


@pytest.mark.slow
def test_multi_col_null_pivot(basic_df, spark_info, memory_leak_check):
    """
    Basic test for PIVOT with multiple columns without a match somewhere.
    """
    query = """
    SELECT * FROM table1
    PIVOT (
        SUM(A) AS SUM_A, AVG(C) AS AVG_C
        FOR (A, B) IN ((1, 4) as COL1, (2, 15) as COL2)
    )
    """
    # set check_dtype=False because of int64 vs Int64 difference
    check_query(
        query,
        basic_df,
        spark_info,
        is_out_distributed=False,
        check_dtype=False,
    )


@pytest.mark.tz_aware
@pytest.mark.slow
def test_tz_aware_pivot(memory_leak_check):
    """
    Tests for using an aggregation function that outputs
    tz-aware data inside of pivot. Note that we cannot
    test using keys yet because this requires tz-aware
    Timestamp literals [BE-4099].
    """
    query = """
    SELECT * FROM table1
    PIVOT (
        MAX(TZ) AS max_val, COUNT(TZ) AS count_val
        FOR (name) IN ('John' as john, 'mike' as mike)
    )
    """
    session_tz = "Poland"
    df = pd.DataFrame(
        {
            "NAME": ["John", "mike", "Luke", "Matthew", None] * 8,
            "TZ": [
                None,
                None,
                pd.Timestamp("2022-1-1", tz=session_tz),
                pd.Timestamp("2028-1-1", tz=session_tz),
                pd.Timestamp("2017-1-1", tz=session_tz),
                None,
                pd.Timestamp("2015-1-1", tz=session_tz),
                pd.Timestamp("2022-1-2", tz=session_tz),
            ]
            * 5,
        }
    )
    ctx = {"TABLE1": df}
    # Compute the expected output
    john_series = df["TZ"][df["NAME"] == "John"]
    john_count = john_series.count()
    john_max = john_series.max()
    mike_series = df["TZ"][df["NAME"] == "mike"]
    mike_count = mike_series.count()
    mike_max = mike_series.max()
    py_output = pd.DataFrame(
        {
            "JOHN_MAX_VAL": john_max,
            "JOHN_COUNT_VAL": john_count,
            "MIKE_MAX_VAL": mike_max,
            "MIKE_COUNT_VAL": mike_count,
        },
        index=pd.RangeIndex(0, 1, 1),
    )
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
        is_out_distributed=False,
        session_tz=session_tz,
    )


@pytest.mark.slow
def test_float_pivot(spark_info, memory_leak_check):
    """
    Basic test for PIVOT that verifies that float values are handling
    with optional types.
    """
    query = """
    select SUM_AMOUNT_2, SUM_AMOUNT_3 from table1
                    PIVOT (sum(amount)
                    for id in (2 as SUM_AMOUNT_2, 3 as SUM_AMOUNT_3) )
    """
    df = pd.DataFrame(
        {
            "ID": [1, 2, 2, 3] * 5,
            "AMOUNT": [0.5, 0.1, 2.3, 11.1, 23] * 4,
        }
    )
    ctx = {"TABLE1": df}
    # set check_dtype=False because of int64 vs Int64 difference
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        is_out_distributed=False,
    )
