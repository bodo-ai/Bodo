"""
Test correctness of SQL queries containing orderby on BodoSQL
"""

import pandas as pd
import pytest

from bodo.tests.timezone_common import representative_tz  # noqa
from bodo.tests.utils import pytest_slow_unless_codegen, temp_config_override
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.fixture(
    params=[
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": [1] * 12,
                    "B": [False, None, True, False] * 3,
                }
            )
        },
        {"TABLE1": pd.DataFrame({"A": ["a"] * 12, "B": [1, 2, 3, 4, 5, 6] * 2})},
        {
            "TABLE1": pd.DataFrame(
                {"A": [0] * 12, "B": ["a", "aa", "aaa", "ab", "b", "hello"] * 2}
            )
        },
    ]
)
def col_a_identical_tables(request):
    """
    Group of tables with identical column A, and varied column B, used for testing groupby
    """
    return request.param


@pytest.mark.slow
@pytest.mark.bodosql_cpp
def test_orderby_numeric_scalar(bodosql_numeric_types, memory_leak_check):
    """tests that orderby works with scalar values in the Select statement"""
    query = "SELECT A, 1, 2, 3, 4 as Y FROM table1 ORDER BY Y, A"
    check_query(
        query,
        bodosql_numeric_types,
        None,
        check_dtype=False,
        check_names=False,
        sort_output=False,
        use_duckdb=True,
    )


@pytest.mark.slow
@pytest.mark.bodosql_cpp
def test_orderby_numeric(bodosql_numeric_types, memory_leak_check):
    """
    Tests orderby works in the simple case for numeric types
    """
    query = """
        SELECT
            A, B, C
        FROM
            table1
        ORDER BY
            A
        """
    query2 = """
        SELECT
            A, B, C
        FROM
            table1
        ORDER BY
            A DESC
        """
    check_query(
        query,
        bodosql_numeric_types,
        None,
        check_dtype=False,
        sort_output=False,
        use_duckdb=True,
    )
    check_query(
        query2,
        bodosql_numeric_types,
        None,
        check_dtype=False,
        sort_output=False,
        use_duckdb=True,
    )


def test_orderby_nullable_numeric(bodosql_nullable_numeric_types, memory_leak_check):
    """
    Tests orderby works in the simple case for nullable numeric types
    """
    query = """
        SELECT
            A, B, C
        FROM
            table1
        ORDER BY
            A
        """
    query2 = """
        SELECT
            A, B, C
        FROM
            table1
        ORDER BY
            A DESC
        """
    output1 = bodosql_nullable_numeric_types["TABLE1"].sort_values(
        by=["A"], ascending=True, na_position="last"
    )
    check_query(
        query,
        bodosql_nullable_numeric_types,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=output1,
    )
    output2 = bodosql_nullable_numeric_types["TABLE1"].sort_values(
        by=["A"], ascending=False, na_position="first"
    )
    check_query(
        query2,
        bodosql_nullable_numeric_types,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=output2,
    )


def test_orderby_bool(bodosql_boolean_types, memory_leak_check):
    """
    Tests orderby works in the simple case for boolean types
    """
    query = """
        SELECT
             A, B, C
        FROM
            table1
        ORDER BY
            A, B, C
        """
    query2 = """
        SELECT
             A, B, C
        FROM
            table1
        ORDER BY
            A, B, C DESC
        """
    output1 = bodosql_boolean_types["TABLE1"].sort_values(
        by=["A", "B", "C"], ascending=True, na_position="last"
    )
    check_query(
        query,
        bodosql_boolean_types,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=output1,
    )
    # Note Pandas doesn't allow passing separate NA position values for A and B yet. As a result we need
    # to write manual code to handle this case.
    output2 = bodosql_boolean_types["TABLE1"].sort_values(
        by=["A", "B", "C"], ascending=[True, True, False], na_position="last"
    )
    output2["C"] = output2.groupby(["A", "B"], dropna=False).transform(
        lambda x: x.sort_values(ascending=False, na_position="first")
    )
    check_query(
        query2,
        bodosql_boolean_types,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=output2,
    )


@pytest.mark.slow
def test_orderby_str(bodosql_string_types, spark_info, memory_leak_check):
    """
    Tests orderby works in the simple case for string types
    Note: We include A to resolve ties.
    """
    query = """
        SELECT
            DISTINCT A, B, C
        FROM
            table1
        ORDER BY
            B, A
        """
    query2 = """
        SELECT
            DISTINCT A, B, C
        FROM
            table1
        ORDER BY
            B DESC, A
        """
    check_query(query, bodosql_string_types, spark_info, sort_output=False)
    check_query(query2, bodosql_string_types, spark_info, sort_output=False)


def test_orderby_binary(bodosql_binary_types, memory_leak_check):
    """
    Tests orderby works in the simple case for binary types
    """
    query = """
        SELECT
            DISTINCT A, B, C
        FROM
            table1
        ORDER BY
            B, A DESC
        """
    # Note Pandas doesn't allow passing separate NA position values for A and B yet. As a result we need
    # to write manual code to handle this case.
    output = bodosql_binary_types["TABLE1"].drop_duplicates()
    output = output.sort_values(
        by=["B", "A"], ascending=[True, False], na_position="last"
    )
    output["A"] = output.groupby(["B", "C"], dropna=False).transform(
        lambda x: x.sort_values(ascending=False, na_position="first")
    )
    check_query(
        query,
        bodosql_binary_types,
        None,
        sort_output=False,
        expected_output=output,
    )


def test_orderby_datetime(bodosql_datetime_types, memory_leak_check):
    """
    Tests orderby works in the simple case for datetime types
    """
    query1 = """
        SELECT
            DISTINCT A, B, C
        FROM
            table1
        ORDER BY
            A, B, C
        """
    query2 = """
        SELECT
            DISTINCT A, B, C
        FROM
            table1
        ORDER BY
            A, B, C DESC
        """
    base_output = bodosql_datetime_types["TABLE1"].drop_duplicates()
    output1 = base_output.sort_values(
        by=["A", "B", "C"], ascending=True, na_position="last"
    )
    check_query(
        query1, bodosql_datetime_types, None, sort_output=False, expected_output=output1
    )
    # Note Pandas doesn't allow passing separate NA position values for A and B yet. As a result we need
    # to write manual code to handle this case.
    output2 = base_output.sort_values(
        by=["A", "B", "C"], ascending=[True, True, False], na_position="last"
    )
    output2["C"] = output2.groupby(["A", "B"], dropna=False).transform(
        lambda x: x.sort_values(ascending=False, na_position="first")
    )
    check_query(
        query2, bodosql_datetime_types, None, sort_output=False, expected_output=output2
    )


def test_orderby_interval(bodosql_interval_types, memory_leak_check):
    """
    Tests orderby works in the simple case for timedelta types
    """
    query1 = """
        SELECT
            DISTINCT A, B, C
        FROM
            table1
        ORDER BY
            A
        """
    query2 = """
        SELECT
            DISTINCT A, B, C
        FROM
            table1
        ORDER BY
            A DESC
        """
    check_query(
        query1,
        bodosql_interval_types,
        None,
        expected_output=bodosql_interval_types["TABLE1"]
        .drop_duplicates()
        .sort_values(by="A", na_position="last"),
    )
    check_query(
        query2,
        bodosql_interval_types,
        None,
        expected_output=bodosql_interval_types["TABLE1"]
        .drop_duplicates()
        .sort_values(by="A", ascending=False, na_position="last"),
    )


@pytest.mark.slow
def test_distinct_orderby(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    Tests orderby and distinct work together as intended
    """
    query = """
        SELECT
            distinct A, B
        FROM
            table1
        ORDER BY
            A
        """
    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, sort_output=False
    )


@pytest.mark.slow
def test_orderby_multiple_cols(col_a_identical_tables, memory_leak_check):
    """
    checks that orderby works correctly when sorting by multiple columns
    """
    query = """
        SELECT
            A, B
        FROM
            table1
        ORDER BY
            A, B
        """
    output = col_a_identical_tables["TABLE1"].sort_values(
        by=["A", "B"], ascending=True, na_position="last"
    )
    check_query(
        query,
        col_a_identical_tables,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=output,
    )


@pytest.fixture(
    params=[
        {
            "TABLE1": pd.DataFrame(
                {"A": [1, 2, None] * 8, "B": [1, 2, 3, None] * 6}, dtype="Int64"
            )
        },
    ]
)
def null_ordering_table(request):
    """
    Tables where null ordering impacts the sort output.
    """
    return request.param


def test_orderby_nulls_defaults(null_ordering_table, spark_info):
    """
    checks that order by null ordering is ASC by default with NULLS LAST
    """
    query = """
        SELECT
            A, B
        FROM
            table1
        ORDER BY
            A, B
        """
    output = null_ordering_table["TABLE1"].sort_values(
        by=["A", "B"], na_position="last"
    )
    check_query(
        query,
        null_ordering_table,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=output,
    )


@pytest.mark.slow
def test_orderby_nulls_defaults_asc(null_ordering_table, memory_leak_check):
    """
    checks that order by null ordering is NULLS LAST for ASC
    """
    query = """
        SELECT
            A, B
        FROM
            table1
        ORDER BY
            A ASC, B
        """
    output = null_ordering_table["TABLE1"].sort_values(
        by=["A", "B"], na_position="last"
    )
    check_query(
        query,
        null_ordering_table,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=output,
    )


@pytest.mark.slow
def test_orderby_nulls_defaults_desc(null_ordering_table, memory_leak_check):
    """
    checks that order by null ordering is NULLS FIRST for DESC
    """
    query = """
        SELECT
            A, B
        FROM
            table1
        ORDER BY
            A DESC, B
    """
    # Note Pandas doesn't allow passing separate NA position values for A and B yet. As a result we need
    # to write manual code to handle this case.
    output = null_ordering_table["TABLE1"].sort_values(
        by=["A", "B"], ascending=[False, True], na_position="first"
    )
    output["B"] = (
        output.groupby(["A"], dropna=False)["B"]
        .transform(lambda x: x.sort_values(ascending=True, na_position="last").values)
        .values
    )
    check_query(
        query,
        null_ordering_table,
        None,
        check_dtype=False,
        sort_output=False,
        expected_output=output,
    )


@pytest.mark.slow
def test_orderby_nulls_first(null_ordering_table, spark_info, memory_leak_check):
    """
    checks that order by null ordering matches with nulls first
    """
    query = """
        SELECT
            A, B
        FROM
            table1
        ORDER BY
            A nulls first, B nulls first
    """
    check_query(
        query,
        null_ordering_table,
        spark_info,
        check_dtype=False,
        sort_output=False,
    )


@pytest.mark.slow
def test_orderby_nulls_last(null_ordering_table, spark_info, memory_leak_check):
    """
    checks that order by null ordering matches with nulls last
    """
    query = """
        SELECT
            A, B
        FROM
            table1
        ORDER BY
            A nulls last, B nulls last
    """
    check_query(
        query,
        null_ordering_table,
        spark_info,
        check_dtype=False,
        sort_output=False,
    )


def test_orderby_nulls_first_last(null_ordering_table, spark_info, memory_leak_check):
    """
    checks that order by null ordering matches with nulls first and last
    """
    query = """
        SELECT
            A, B
        FROM
            table1
        ORDER BY
            A nulls first, B nulls last
    """
    check_query(
        query,
        null_ordering_table,
        spark_info,
        check_dtype=False,
        sort_output=False,
    )


def test_orderby_tz_aware(representative_tz, memory_leak_check):
    """
    Test various ORDER BY operations on tz-aware data
    """
    df = pd.DataFrame(
        {
            "A": [
                pd.Timestamp("2022-1-1", tz=representative_tz),
                pd.Timestamp("2022-1-3", tz=representative_tz),
                None,
                pd.Timestamp("2023-11-13", tz=representative_tz),
                pd.Timestamp("2021-11-4", tz=representative_tz),
            ]
            * 2,
            "B": [
                None,
                pd.Timestamp("2021-1-1", tz=representative_tz),
                pd.Timestamp("2021-1-2", tz=representative_tz),
                pd.Timestamp("2024-1-1", tz=representative_tz),
                pd.Timestamp("2024-1-3", tz=representative_tz),
                pd.Timestamp("2019-1-1", tz=representative_tz),
                pd.Timestamp("2023-1-3", tz=representative_tz),
                pd.Timestamp("2015-1-1", tz=representative_tz),
                pd.Timestamp("2011-1-3", tz=representative_tz),
                None,
            ],
            # This is just a Data Column
            "C": pd.date_range(
                "2022/1/1", freq="13min", periods=10, tz=representative_tz, unit="ns"
            ),
        }
    )
    ctx = {"TABLE1": df}
    # NOTE: Pandas doesn't support using different NULLS FIRST or NULLS LAST
    # per column, so we will manually convert NULL to a different type.
    large_timestamp = pd.Timestamp("2050-1-1", tz=representative_tz)
    small_timestamp = pd.Timestamp("1970-1-1", tz=representative_tz)

    query1 = """
        Select *
        FROM table1
        ORDER BY
            A DESC nulls last,
            B ASC nulls first
    """
    py_output = df.fillna(small_timestamp)
    py_output = py_output.sort_values(["A", "B"], ascending=[False, True])
    # Reset small_timestamp to None
    col_a_nas = py_output["A"] == small_timestamp
    col_b_nas = py_output["B"] == small_timestamp
    py_output["A"] = py_output["A"].where(~col_a_nas, None)
    py_output["B"] = py_output["B"].where(~col_b_nas, None)
    check_query(
        query1,
        ctx,
        None,
        sort_output=False,
        expected_output=py_output,
        session_tz=representative_tz,
    )

    query2 = """
        Select *
        FROM table1
        ORDER BY
            A ASC nulls last,
            B DESC nulls first
    """
    py_output = df.fillna(large_timestamp)
    py_output = py_output.sort_values(["A", "B"], ascending=[True, False])
    # Reset small_timestamp to None
    col_a_nas = py_output["A"] == large_timestamp
    col_b_nas = py_output["B"] == large_timestamp
    py_output["A"] = py_output["A"].where(~col_a_nas, None)
    py_output["B"] = py_output["B"].where(~col_b_nas, None)
    check_query(
        query2,
        ctx,
        None,
        sort_output=False,
        expected_output=py_output,
        session_tz=representative_tz,
    )


@pytest.mark.parametrize(
    "query, answer",
    [
        (
            "Select A from TABLE1 order by A",
            pd.array([None, None, 1, 1, 2, 2, 3, 4, 5], dtype="Int64"),
        ),
        (
            "Select A from TABLE1 order by A ASC",
            pd.array([None, None, 1, 1, 2, 2, 3, 4, 5], dtype="Int64"),
        ),
        (
            "Select A from TABLE1 order by A DESC",
            pd.array([5, 4, 3, 2, 2, 1, 1, None, None], dtype="Int64"),
        ),
    ],
)
def test_orderby_spark_style(query, answer, memory_leak_check):
    """
    Test that with BODO_SQL_STYLE set to Spark we match spark order by
    rules.
    """

    with temp_config_override("bodo_sql_style", "SPARK"):
        df = pd.DataFrame(
            {
                "A": pd.array([1, 2, 3, 4, None, 5, None, 1, 2], dtype="Int64"),
            }
        )
        check_query(
            query,
            {"TABLE1": df},
            None,
            expected_output=pd.DataFrame({"A": answer}),
            sort_output=False,
            check_dtype=False,
        )
