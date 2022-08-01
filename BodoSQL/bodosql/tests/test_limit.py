import pytest
from bodosql.tests.utils import check_query


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
