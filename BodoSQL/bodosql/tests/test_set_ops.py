"""
Test correctness of SQL set like operations, namely UNION, INTERSECT, and EXCEPT
"""

from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from bodo.tests.timezone_common import representative_tz  # noqa
from bodo.tests.utils import pytest_slow_unless_codegen
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.fixture(
    params=[
        pytest.param("UNION", id="union_op"),
        pytest.param("UNION DISTINCT", id="union_distinct", marks=pytest.mark.slow),
        pytest.param("UNION ALL", id="union_all"),
    ]
)
def union_cmds(request) -> str:
    return request.param


@pytest.fixture(
    params=[
        pytest.param("INTERSECT", id="intersect_op"),
        pytest.param(
            "INTERSECT DISTINCT", id="intersect_distinct", marks=pytest.mark.slow
        ),
        pytest.param("INTERSECT ALL", id="intersect_all", marks=pytest.mark.slow),
    ]
)
def intersect_cmds(request) -> str:
    return request.param


@pytest.fixture(
    params=[
        pytest.param("EXCEPT", id="except_op"),
        pytest.param("MINUS", id="minus_op", marks=pytest.mark.slow),
        pytest.param("EXCEPT ALL", id="except_all"),
        pytest.param("MINUS ALL", id="minus_all", marks=pytest.mark.slow),
    ]
)
def except_cmds(request) -> str:
    return request.param


@pytest.fixture(
    params=[
        pytest.param("UNION", id="union_op"),
        pytest.param("UNION DISTINCT", id="union_distinct", marks=pytest.mark.slow),
        pytest.param("UNION ALL", id="union_all"),
        pytest.param("INTERSECT", id="intersect_op"),
        pytest.param(
            "INTERSECT DISTINCT", id="intersect_distinct", marks=pytest.mark.slow
        ),
        pytest.param("INTERSECT ALL", id="intersect_all", marks=pytest.mark.slow),
        pytest.param("EXCEPT", id="except_op", marks=pytest.mark.slow),
        pytest.param("MINUS", id="minus_op", marks=pytest.mark.slow),
        pytest.param("EXCEPT ALL", id="except_all"),
        pytest.param("MINUS ALL", id="minus_all", marks=pytest.mark.slow),
    ]
)
def set_ops_cmds(request) -> str:
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            {
                "TABLE1": pd.DataFrame(
                    {
                        "A": pd.Series([1, 2, 3, None] * 3),
                        "B": pd.Series([4, None, 2, 3] * 3),
                        "C": pd.Series([None, 1, 2, 3] * 3),
                    },
                    dtype="Int64",
                )
            },
            id="int",
        ),
        pytest.param(
            {
                "TABLE1": pd.DataFrame(
                    {
                        "A": pd.Series(["a", "b", "cc", None] * 3),
                        "B": pd.Series(["d", None, "bb", "c"] * 3),
                        "C": pd.Series([None, "aa", "b", "c"] * 3),
                    },
                    dtype="string",
                )
            },
            id="string",
        ),
    ]
)
def null_set_dfs(request):
    return request.param


@pytest.fixture(
    params=[
        {
            # What Spark uses to demonstrate set operators:
            # https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-setops.html
            "TABLE1": pd.DataFrame({"A": [3, 1, 2, 2, 3, 4]}),
            "TABLE2": pd.DataFrame({"B": [5, 1, 2, 2]}),
        },
        {
            # Testing how duplicates are handled
            "TABLE1": pd.DataFrame({"A": [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5]}),
            "TABLE2": pd.DataFrame({"B": [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 6]}),
        },
    ]
)
def set_ops_dfs(request):
    return request.param


def make_tz_aware_df(tz):
    """
    Make a DataFrame of tz-aware timestamps with the given timezone.
    """
    # Note: B's and A's will overlap.
    tz_aware_data = {
        "A": list(
            pd.date_range(start="1/1/2022", freq="4D7h", periods=30, tz=tz, unit="ns")
        )
        + [None] * 4,
        "B": [None] * 14
        + list(
            pd.date_range(start="1/1/2022", freq="12D21h", periods=20, tz=tz, unit="ns")
        ),
    }
    return pd.DataFrame(data=tz_aware_data)


# ===== Set Ops Tests


@pytest.mark.parametrize(
    "numeric_dfs",
    [
        # Integer
        {
            "TABLE1": pd.DataFrame({"A": pd.Series([3, 1, 2, 2, 3, 4], dtype="int32")}),
            "TABLE2": pd.DataFrame({"B": pd.Series([5, 1, 2, 2], dtype="Int8")}),
        },
        # Float
        {
            "TABLE1": pd.DataFrame(
                {"A": pd.Series([3.0, 1.0, 2.0, 2.0, 3.0, 4.0], dtype="float64")}
            ),
            "TABLE2": pd.DataFrame(
                {"B": pd.Series([5.0, 1.0, 2.0, 2.0], dtype="Float32")}
            ),
        },
        # Float <-> Integer
        {
            "TABLE1": pd.DataFrame(
                {"A": pd.Series([3.0, 1.0, 2.0, 2.0, 3.0, 4.0], dtype="float64")}
            ),
            "TABLE2": pd.DataFrame({"B": pd.Series([5, 1, 2, 2], dtype="Int64")}),
        },
        # Decimal
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": np.array(
                        [
                            Decimal("1.6"),
                            None,
                            Decimal("-0.222"),
                            Decimal("1111.316"),
                            Decimal("1234.00046"),
                            Decimal("5.1"),
                            Decimal("-11131.0056"),
                            Decimal("0.0"),
                        ]
                    )
                }
            ),
            "TABLE2": pd.DataFrame(
                {
                    "B": np.array(
                        [
                            Decimal("1.6"),
                            None,
                            Decimal("200.78"),
                            Decimal("-0.222"),
                            Decimal("-15.78"),
                            Decimal("1111.316"),
                        ]
                    )
                }
            ),
        },
        # Decimal <-> Float
        {
            "TABLE1": pd.DataFrame(
                {"A": pd.Series([3.0, 1.0, 2.0, 2.0, 3.0, 4.0], dtype="float64")}
            ),
            "TABLE2": pd.DataFrame(
                {
                    "B": np.array(
                        [
                            Decimal("1.6"),
                            None,
                            Decimal("200.78"),
                            Decimal("3.000"),
                            Decimal("-15.78"),
                            Decimal("1111.316"),
                        ]
                    )
                }
            ),
        },
        # Decimal <-> Integer
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": np.array(
                        [
                            Decimal("1.0"),
                            None,
                            Decimal("200.78"),
                            Decimal("-3.000"),
                            Decimal("-15.78"),
                            Decimal("1111.316"),
                        ]
                    )
                }
            ),
            "TABLE2": pd.DataFrame(
                {"B": pd.Series([5, 1, 2, pd.NA, 2], dtype="Int64")}
            ),
        },
    ],
)
def test_union_numeric_cols(numeric_dfs, union_cmds, spark_info, memory_leak_check):
    """
    Test that UNION works for a single numeric column

    Union Casting Logic is based on investigation from:
    https://bodo.atlassian.net/wiki/spaces/B/pages/1474134034/Numeric+Casting+Investigation+for+Union
    but assuming:
    - float == Snowflake Float != Snowflake Number
        so can never cast float => decimal
    - Truncation of float + decimal => float allowed
    """
    query = f"SELECT A FROM table1 {union_cmds} SELECT B FROM table2"
    check_query(query, numeric_dfs, spark_info, check_dtype=False, check_names=False)


@pytest.mark.slow
def test_set_ops_numeric_cols(set_ops_dfs, set_ops_cmds, spark_info, memory_leak_check):
    """
    Test that set ops work for a single numeric column
    """
    query = f"SELECT A FROM table1 {set_ops_cmds} SELECT B FROM table2"
    check_query(query, set_ops_dfs, spark_info, check_dtype=False)


def test_set_ops_nullable_many_cols(
    null_set_dfs, set_ops_cmds, spark_info, memory_leak_check
):
    """
    Test that set ops work for many nullable columns
    """
    query = f"SELECT A, B, C FROM table1 {set_ops_cmds} SELECT C, C, B FROM table1"
    check_query(query, null_set_dfs, spark_info, check_dtype=False)


@pytest.mark.parametrize(
    "query_cmd, py_output",
    [
        pytest.param(
            "UNION",
            pd.DataFrame(
                {
                    "A": pd.Series([1, 2, 3, 4, None], dtype="Int64"),
                    "B": pd.Series([1.1, 2.7, 3.4, 110.3, None], dtype="Float64"),
                    "C": ["A", None, "c", "recall", None],
                }
            ),
            id="union",
        ),
        pytest.param(
            "INTERSECT",
            pd.DataFrame(
                {
                    "A": pd.Series([None], dtype="Int64"),
                    "B": pd.Series([None], dtype="Float64"),
                    "C": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                }
            ),
            id="intersect",
            marks=pytest.mark.skip("[BE-4320] Support null literals in intersect"),
        ),
        pytest.param(
            "EXCEPT",
            pd.DataFrame(
                {
                    "A": pd.Series([1, 2, 3, 4], dtype="Int64"),
                    "B": pd.Series([1.1, 2.7, 3.4, 110.3], dtype="Float64"),
                    "C": ["A", None, "c", "recall"],
                }
            ),
            id="except",
            marks=pytest.mark.skip("[BE-4320] Support null literals in except"),
        ),
    ],
)
def test_set_ops_null_literals(query_cmd, py_output, memory_leak_check):
    """
    Test for [BE-4320], checks that set ops works for various null literals
    """
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, None] * 3,
            "B": [1.1, 2.7, 3.4, 110.3, None] * 3,
            "C": ["A", None, "c", "recall", None] * 3,
        }
    )
    ctx = {"TABLE1": df}
    query = f"(SELECT A, B, C from table1) {query_cmd} (SELECT null as A, null as B, null as C)"
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.slow
def test_set_ops_string_cols(
    bodosql_string_types, set_ops_cmds, spark_info, memory_leak_check
):
    """
    Test that set ops work for string columns
    """
    query = f"SELECT A FROM table1 {set_ops_cmds} SELECT B FROM table1"
    check_query(query, bodosql_string_types, spark_info, check_dtype=False)


def test_set_ops_datetime_cols(
    bodosql_datetime_types, set_ops_cmds, spark_info, memory_leak_check
):
    """
    Test that set ops work for datetime columns
    """
    query = f"SELECT A FROM table1 {set_ops_cmds} SELECT B FROM table1"
    check_query(query, bodosql_datetime_types, spark_info, use_duckdb=True)


@pytest.mark.slow
def test_set_ops_binary_cols(
    bodosql_binary_types, set_ops_cmds, spark_info, memory_leak_check
):
    """
    Test that set ops work for binary columns
    """
    query = f"SELECT A FROM table1 {set_ops_cmds} SELECT B FROM table1"
    check_query(query, bodosql_binary_types, spark_info, check_dtype=False)


# ===== String Restrictions


def test_union_string_restrictions(
    bodosql_string_types, union_cmds, spark_info, memory_leak_check
):
    """
    Test that UNION [ALL/DISTINCT] works for string columns when used with restrictions
    """
    query = (
        f"(SELECT A, B, C FROM table1 WHERE LENGTH(A) = 5) {union_cmds} "
        f"(SELECT CASE WHEN LENGTH(B) > 4 THEN B END, C, A FROM table1 ORDER BY C) {union_cmds} "
        f"(SELECT B, C, A FROM table1 WHERE LENGTH(A) = 5 AND LENGTH(B) > 3 ORDER BY B, A LIMIT 2) {union_cmds} "
        f"(SELECT C, B, A FROM table1)"
    )
    check_query(
        query, bodosql_string_types, spark_info, check_names=False, check_dtype=False
    )


@pytest.mark.slow
def test_intersect_string_restrictions(
    bodosql_string_types, intersect_cmds, spark_info, memory_leak_check
):
    """
    Test that INTERSECT [ALL/DISTINCT] works for string columns when used with restrictions
    """
    query = (
        f"(SELECT CASE WHEN LENGTH(B) > 4 THEN B END, C, A FROM table1 ORDER BY C) {intersect_cmds} "
        # Note that we need the order by "A" in this test to break ties.
        f"(SELECT B, C, A FROM table1 WHERE LENGTH(A) = 5 AND LENGTH(B) > 3 ORDER BY B, A LIMIT 2) {intersect_cmds} "
        f"(SELECT B, C, A FROM table1)"
    )
    check_query(
        query, bodosql_string_types, spark_info, check_names=False, check_dtype=False
    )


def test_except_string_restrictions(
    bodosql_string_types, except_cmds, spark_info, memory_leak_check
):
    """
    Test that EXCEPT/MINUS [ALL] works for string columns when used with restrictions
    """
    query = (
        f"(SELECT CASE WHEN LENGTH(B) > 4 THEN B END, C, A FROM table1 ORDER BY C) {except_cmds} "
        f"(SELECT B, C, A FROM table1 WHERE LENGTH(A) = 5 AND LENGTH(B) > 3 ORDER BY B, A LIMIT 2) "
    )
    check_query(
        query, bodosql_string_types, spark_info, check_names=False, check_dtype=False
    )


# ===== Timezone-Aware Cols


@pytest.mark.slow
def test_union_tz_aware_cols(union_cmds, representative_tz, memory_leak_check):
    """
    Tests that UNION [ALL/DISTINCT] works for tz_aware columns
    """
    df = make_tz_aware_df(representative_tz)
    py_output = pd.DataFrame({"A": pd.concat((df["A"], df["B"]))})
    if "ALL" in union_cmds:
        # For UNION ALL, number of NA's should be A_na + B_na
        num_na = df["A"].isna().sum() + df["B"].isna().sum()
        py_output = py_output.dropna()
        py_output = pd.DataFrame({"A": list(py_output["A"]) + [None] * num_na})
    else:
        # Drop duplicates for UNION [DISTINCT]
        py_output = py_output.drop_duplicates()

    ctx = {"TABLE1": df}
    query = f"SELECT A FROM table1 {union_cmds} SELECT B FROM table1"
    check_query(
        query, ctx, None, expected_output=py_output, session_tz=representative_tz
    )


def test_intersect_tz_aware_cols(intersect_cmds, representative_tz, memory_leak_check):
    """
    Test that INTERSECT [ALL/DISTINCT] works for tz_aware columns
    """
    df = make_tz_aware_df(representative_tz)
    py_output = df[["A"]].merge(df[["B"]].rename(columns={"B": "A"}), on="A")
    if "ALL" in intersect_cmds:
        # For INTERSECT ALL, number of NA's should be min(A_na, B_na)
        num_na = min(df["A"].isna().sum(), df["B"].isna().sum())
        py_output = py_output.dropna()
        py_output = pd.DataFrame({"A": list(py_output["A"]) + [None] * num_na})
    else:
        # Drop duplicates for INTERSECT [DISTINCT]
        py_output = py_output.drop_duplicates()

    ctx = {"TABLE1": df}
    query = f"SELECT A FROM table1 {intersect_cmds} SELECT B FROM table1"
    check_query(
        query, ctx, None, expected_output=py_output, session_tz=representative_tz
    )


@pytest.mark.slow
def test_except_tz_aware_cols(except_cmds, representative_tz, memory_leak_check):
    """
    Tests that EXCEPT/MINUS [ALL] works for tz_aware columns
    """
    df = make_tz_aware_df(representative_tz)
    py_output = pd.DataFrame({"A": list(set(df["A"]).difference(set(df["B"])))})
    if "ALL" in except_cmds:
        # For EXCEPT/MINUS ALL, number of NA's should be max(A_na - B_na, 0)
        num_na = max(df["A"].isna().sum() - df["B"].isna().sum(), 0)
        py_output = py_output.dropna()
        py_output = pd.DataFrame({"A": list(py_output["A"]) + [None] * num_na})
    else:
        # Drop duplicates for EXCEPT/MINUS
        py_output = py_output.drop_duplicates()

    ctx = {"TABLE1": df}
    query = f"SELECT A FROM table1 {except_cmds} SELECT B FROM table1"
    check_query(
        query, ctx, None, expected_output=py_output, session_tz=representative_tz
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "query, py_output",
    [
        # [BS-381] These queries are not valid for spark
        pytest.param(
            "SELECT 1,2 UNION SELECT 1,2",
            pd.DataFrame({"A": [1], "B": [2]}),
            id="union_op_same",
        ),
        pytest.param(
            "SELECT 1,2 UNION DISTINCT SELECT 1,3",
            pd.DataFrame({"A": [1, 1], "B": [2, 3]}),
            id="union_distinct_diff",
        ),
        pytest.param(
            "SELECT 1,2 UNION ALL SELECT 1,2",
            pd.DataFrame({"A": [1, 1], "B": [2, 2]}),
            id="union_all_same",
        ),
        pytest.param(
            "(SELECT 1,2 UNION ALL SELECT 1,2) UNION ALL SELECT 1,2",
            pd.DataFrame({"A": [1, 1, 1], "B": [2, 2, 2]}),
            id="union_all_dup",
        ),
        pytest.param(
            "SELECT 1,2 INTERSECT SELECT 1,2",
            pd.DataFrame({"A": [1], "B": [2]}),
            id="intersect_op_same",
        ),
        pytest.param(
            "SELECT 1,2 INTERSECT DISTINCT SELECT 2,3",
            pd.DataFrame({"A": [], "B": []}),
            id="intersect_distinct_diff",
        ),
        pytest.param(
            "SELECT 1,2 INTERSECT ALL SELECT 1,2",
            pd.DataFrame({"A": [1], "B": [2]}),
            id="intersect_all_same",
        ),
        pytest.param(
            "(SELECT 1,2 UNION ALL SELECT 1,2) INTERSECT ALL (SELECT 1,2 UNION ALL SELECT 1,2)",
            pd.DataFrame({"A": [1, 1], "B": [2, 2]}),
            id="intersect_all_dup",
        ),
        pytest.param(
            "SELECT 1,2 EXCEPT SELECT 1,2",
            pd.DataFrame({"A": [], "B": []}),
            id="except_op_same",
        ),
        pytest.param(
            "SELECT 1,2 MINUS SELECT 2,3",
            pd.DataFrame({"A": [1], "B": [2]}),
            id="minus_op_diff",
        ),
        pytest.param(
            "SELECT 1,2 EXCEPT ALL SELECT 1,2",
            pd.DataFrame({"A": [], "B": []}),
            id="except_all_same",
        ),
        pytest.param(
            "(SELECT 1,2 UNION ALL SELECT 1,2) MINUS ALL SELECT 1,2",
            pd.DataFrame({"A": [1], "B": [2]}),
            id="minus_all_dup",
        ),
    ],
)
def test_set_ops_scalars(query, py_output, memory_leak_check):
    """
    Tests that set ops work for scalars
    """
    check_query(
        query,
        {},
        None,
        check_names=False,
        expected_output=py_output,
        check_dtype=False,
    )
