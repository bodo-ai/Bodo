import pandas as pd
import pytest
from bodosql.tests.utils import check_query

from bodo.tests.timezone_common import representative_tz  # noqa


def test_limit_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """test queries with limit"""
    query = "select B,C from table1 limit 4"
    check_query(query, bodosql_numeric_types, spark_info, check_dtype=False)


def test_limit_offset_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """test queries with limit and offset. Here offset=1 and limit=4"""
    query = "select B,C from table1 limit 1, 4"
    # Spark doesn't support offset so use an expected output
    expected_output = bodosql_numeric_types["table1"].iloc[1:5, [1, 2]]
    check_query(
        query, bodosql_numeric_types, spark_info, expected_output=expected_output
    )


@pytest.mark.slow
def test_limit_offset_keyword(basic_df, spark_info, memory_leak_check):
    """test queries with limit and offset. Here offset=1 and limit=4"""
    query = "select B,C from table1 limit 4 offset 1"
    # Spark doesn't support offset so use an expected output
    expected_output = basic_df["table1"].iloc[1:5, [1, 2]]
    check_query(query, basic_df, spark_info, expected_output=expected_output)


def test_limit_tz_aware(representative_tz, memory_leak_check):
    """test limit and variants with tz_aware data"""
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                "2022/1/1", periods=30, freq="6D5H", tz=representative_tz
            ),
            "B": pd.date_range("2022/1/1", periods=30, freq="11D", tz="UTC"),
            "C": pd.date_range(
                "2022/1/1",
                periods=30,
                freq="1H",
            ),
        }
    )
    ctx = {"table1": df}
    query = "select A, C from table1 limit 10"
    expected_output = df[["A", "C"]].head(10)
    check_query(query, ctx, None, expected_output=expected_output)
    query = "select A, C from table1 limit 4 offset 1"
    expected_output = df.iloc[1:5, [0, 2]]
    check_query(query, ctx, None, expected_output=expected_output)
