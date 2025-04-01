"""
Test correctness of SQL ANY and ALL clauses.
"""

import pandas as pd
import pytest

from bodo.tests.utils import pytest_slow_unless_codegen
from bodosql.tests.string_ops_common import *  # noqa
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.fixture(
    params=[
        pytest.param("ANY"),
        pytest.param("ALL"),
    ]
)
def some_any_all(request):
    """Returns a string that is either SOME ANY or ALL"""
    return request.param


def test_some_any_all_numeric_non_null_tuples(
    basic_df, some_any_all, comparison_ops, spark_info, memory_leak_check
):
    """Tests that SOME and ALL work with numeric value tuples"""

    query = f"SELECT A FROM table1 WHERE A {comparison_ops} {some_any_all} (1, 2)"

    if some_any_all == "ANY" or some_any_all == "SOME":
        spark_query = (
            f"SELECT A FROM table1 WHERE A {comparison_ops} 1 OR A {comparison_ops} 2 "
        )
    else:
        spark_query = (
            f"SELECT A FROM table1 WHERE A {comparison_ops} 1 AND A {comparison_ops} 2 "
        )

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_some_any_all_string_non_null_tuples(
    bodosql_string_types, some_any_all, comparison_ops, spark_info, memory_leak_check
):
    """Tests that some, any, and all work on non null String tuples"""
    query = f"""SELECT A FROM table1 WHERE A {comparison_ops} {some_any_all} ('A', 'B', 'C')"""

    if some_any_all == "ANY" or some_any_all == "SOME":
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} 'A' OR A {comparison_ops} 'B' OR A {comparison_ops} 'C' "
    else:
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} 'A' AND A {comparison_ops} 'B' AND A {comparison_ops} 'C' "

    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_some_any_all_datetime_non_null_tuples(
    bodosql_datetime_types, some_any_all, comparison_ops, spark_info, memory_leak_check
):
    """Tests that some, any, and all work on Timestamp tuples"""
    query = f"""
        SELECT
            A
        FROM
            table1
        WHERE
            A {comparison_ops} {some_any_all} (TIMESTAMP '2011-01-01', TIMESTAMP '1998-08-10', TIMESTAMP '2008-12-13')
        """

    if some_any_all == "ANY" or some_any_all == "SOME":
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} TIMESTAMP '2011-01-01' OR A {comparison_ops} TIMESTAMP '1998-08-10' OR A {comparison_ops} TIMESTAMP '2008-12-13'"
    else:
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} TIMESTAMP '2011-01-01' AND A {comparison_ops} TIMESTAMP '1998-08-10' AND A {comparison_ops} TIMESTAMP '2008-12-13'"

    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_some_any_all_interval_non_null_tuples(
    bodosql_interval_types, some_any_all, comparison_ops, spark_info, memory_leak_check
):
    """Tests that the basic comparison operators work with Timedelta data within the same table"""
    query = f""" SELECT A FROM table1 WHERE A {comparison_ops} {some_any_all} (INTERVAL 1 HOUR, INTERVAL 2 SECOND, INTERVAL -3 DAY)"""

    # Spark casts interval types to bigint, doing comparisons to bigint equivalents of the above intervals to avoid typing errors.
    if some_any_all == "ANY" or some_any_all == "SOME":
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} 3600000000000 OR A {comparison_ops} 2000000000 OR A {comparison_ops} -259200000000000"
    else:
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} 3600000000000 AND A {comparison_ops} 2000000000 AND A {comparison_ops} -259200000000000"

    # Convert Spark input to int since it doesn't support timedelta nulls properly
    df = bodosql_interval_types["TABLE1"].copy()
    for col in df.columns.copy():
        df[col] = df[col].astype(np.int64)

    check_query(
        query,
        bodosql_interval_types,
        spark_info,
        check_dtype=False,
        convert_columns_timedelta=["A"],
        equivalent_spark_query=spark_query,
        spark_dataframe_dict={"TABLE1": df},
    )


@pytest.mark.slow
def test_some_any_all_null_tuples(
    bodosql_nullable_numeric_types,
    some_any_all,
    comparison_ops,
    spark_info,
    memory_leak_check,
):
    """Tests that SOME and ALL work with numeric value tuples"""

    query = f"SELECT A FROM table1 WHERE A {comparison_ops} {some_any_all} (1, NULL)"

    if some_any_all == "ANY" or some_any_all == "SOME":
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} 1 OR A {comparison_ops} NULL "
    else:
        spark_query = f"SELECT A FROM table1 WHERE A {comparison_ops} 1 AND A {comparison_ops} NULL "

    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_some_any_all_subquery(
    simple_join_fixture, some_any_all, comparison_ops, memory_leak_check
):
    """Tests that SOME and ALL work with subqueries"""

    if comparison_ops == "<=>":
        pytest.skip("<=> is not compatible with subqueries")

    expected_output = {
        ("ANY", "="): lambda: pd.DataFrame(
            {
                "A": pd.Series([2, 3], dtype="Int64"),
                "D": pd.Series([5, 6], dtype="Int64"),
            }
        ),
        ("ALL", "="): lambda: pd.DataFrame(
            {
                "A": pd.Series([], dtype="Int64"),
                "D": pd.Series([], dtype="Int64"),
            }
        ),
        ("ANY", "<>"): lambda: pd.DataFrame(
            {
                "A": pd.Series([1, 2, 3], dtype="Int64"),
                "D": pd.Series([4, 5, 6], dtype="Int64"),
            }
        ),
        ("ALL", "<>"): lambda: pd.DataFrame(
            {
                "A": pd.Series([1], dtype="Int64"),
                "D": pd.Series([4], dtype="Int64"),
            }
        ),
        ("ANY", "!="): lambda: pd.DataFrame(
            {
                "A": pd.Series([1, 2, 3], dtype="Int64"),
                "D": pd.Series([4, 5, 6], dtype="Int64"),
            }
        ),
        ("ALL", "!="): lambda: pd.DataFrame(
            {
                "A": pd.Series([1], dtype="Int64"),
                "D": pd.Series([4], dtype="Int64"),
            }
        ),
        ("ANY", "<="): lambda: pd.DataFrame(
            {
                "A": pd.Series([1, 2, 3], dtype="Int64"),
                "D": pd.Series([4, 5, 6], dtype="Int64"),
            }
        ),
        ("ALL", "<="): lambda: pd.DataFrame(
            {
                "A": pd.Series([1, 2], dtype="Int64"),
                "D": pd.Series([4, 5], dtype="Int64"),
            }
        ),
        ("ANY", "<"): lambda: pd.DataFrame(
            {
                "A": pd.Series([1, 2, 3], dtype="Int64"),
                "D": pd.Series([4, 5, 6], dtype="Int64"),
            }
        ),
        ("ALL", "<"): lambda: pd.DataFrame(
            {
                "A": pd.Series([1], dtype="Int64"),
                "D": pd.Series([4], dtype="Int64"),
            }
        ),
        ("ANY", ">="): lambda: pd.DataFrame(
            {
                "A": pd.Series([2, 3], dtype="Int64"),
                "D": pd.Series([5, 6], dtype="Int64"),
            }
        ),
        ("ALL", ">="): lambda: pd.DataFrame(
            {
                "A": pd.Series([], dtype="Int64"),
                "D": pd.Series([], dtype="Int64"),
            }
        ),
        ("ANY", ">"): lambda: pd.DataFrame(
            {
                "A": pd.Series([3], dtype="Int64"),
                "D": pd.Series([6], dtype="Int64"),
            }
        ),
        ("ALL", ">"): lambda: pd.DataFrame(
            {
                "A": pd.Series([], dtype="Int64"),
                "D": pd.Series([], dtype="Int64"),
            }
        ),
    }[some_any_all, comparison_ops]()

    query = f"SELECT A, D FROM table1 WHERE (A + 2) {comparison_ops} {some_any_all} (SELECT C FROM table2)"
    check_query(
        query,
        simple_join_fixture,
        spark=None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )
