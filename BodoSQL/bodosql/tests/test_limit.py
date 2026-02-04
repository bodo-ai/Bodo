import pandas as pd
import pytest

from bodo.tests.timezone_common import representative_tz  # noqa
from bodosql.tests.utils import check_query


def test_limit_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """test queries with limit"""
    query = "select B,C from table1 limit 4"
    check_query(query, bodosql_numeric_types, spark_info, check_dtype=False)


def test_limit_offset_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """test queries with limit and offset. Here offset=1 and limit=4"""
    query = "select B,C from table1 limit 1, 4"
    # Spark doesn't support offset so use an expected output
    expected_output = bodosql_numeric_types["TABLE1"].iloc[1:5, [1, 2]]
    check_query(
        query, bodosql_numeric_types, spark_info, expected_output=expected_output
    )


@pytest.mark.slow
def test_limit_offset_keyword(basic_df, memory_leak_check):
    """test queries with limit and offset. Here offset=1 and limit=4"""
    query = "select B,C from table1 limit 4 offset 1"
    # Spark doesn't support offset so use an expected output
    expected_output = basic_df["TABLE1"].iloc[1:5, [1, 2]]
    check_query(query, basic_df, None, expected_output=expected_output)


@pytest.mark.slow
def test_fetch(basic_df, memory_leak_check):
    """test queries with fetch"""
    query1 = "select B,C from table1 FETCH FIRST 4 ROW"
    query2 = "select B,C from table1 FETCH FIRST 4 ROWS ONLY"
    # Next and first mean the same thing
    query3 = "select B,C from table1 FETCH NEXT 4 ROWS"
    # Generate an expected output
    expected_output1 = basic_df["TABLE1"].iloc[:4, [1, 2]]
    check_query(query1, basic_df, None, expected_output=expected_output1)
    check_query(query2, basic_df, None, expected_output=expected_output1)
    check_query(query3, basic_df, None, expected_output=expected_output1)
    # Test adding an offset
    query4 = "select B,C from table1 OFFSET 1 ROWS FETCH FIRST 4 ROW"
    query5 = "select B,C from table1 OFFSET 1 ROW FETCH FIRST 4 ROWS ONLY"
    # Next and first mean the same thing
    query6 = "select B,C from table1 OFFSET 1 ROWS FETCH NEXT 4 ROWS"
    # Generate an expected output
    expected_output2 = basic_df["TABLE1"].iloc[1:5, [1, 2]]
    check_query(query4, basic_df, None, expected_output=expected_output2)
    check_query(query5, basic_df, None, expected_output=expected_output2)
    check_query(query6, basic_df, None, expected_output=expected_output2)


@pytest.mark.slow
def test_top(basic_df, memory_leak_check):
    """test queries with top"""
    query1 = "select TOP 4 B,C from table1"
    # Generate an expected output
    expected_output1 = basic_df["TABLE1"].iloc[:4, [1, 2]]
    check_query(query1, basic_df, None, expected_output=expected_output1)


def test_limit_tz_aware(representative_tz, memory_leak_check):
    """test limit and variants with tz_aware data"""
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                "2022/1/1", periods=30, freq="6D5h", tz=representative_tz, unit="ns"
            ),
            "B": pd.date_range("2022/1/1", periods=30, freq="11D", tz="UTC", unit="ns"),
            "C": pd.date_range(
                "2022/1/1",
                periods=30,
                freq="1h",
                unit="ns",
            ),
        }
    )
    ctx = {"TABLE1": df}
    query = "select A, C from table1 limit 10"
    expected_output = df[["A", "C"]].head(10)
    check_query(
        query, ctx, None, expected_output=expected_output, session_tz=representative_tz
    )
    query = "select A, C from table1 limit 4 offset 1"
    expected_output = df.iloc[1:5, [0, 2]]
    check_query(
        query, ctx, None, expected_output=expected_output, session_tz=representative_tz
    )
