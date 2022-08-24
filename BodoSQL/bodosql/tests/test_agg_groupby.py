# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL aggregation operations with groupby on BodoSQL
"""
import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture
def grouped_dfs():
    """
    A dataFrame larger amounts of data per group.
    This ensures we get meaningful results with
    groupby functions.
    """
    return {
        "table1": pd.DataFrame(
            {
                "A": [0, 1, 2, 4] * 50,
                "B": np.arange(200),
                "C": [1.5, 2.24, 3.52, 4.521, 5.2353252, -7.3, 23, -0.341341] * 25,
            }
        )
    }


@pytest.mark.slow
def test_agg_numeric(
    bodosql_numeric_types, numeric_agg_builtin_funcs, spark_info, memory_leak_check
):
    """test aggregation calls in queries"""
    # Skipping VAR_POP and STDDEV_POP with groupby due to [BE-910]
    if (
        numeric_agg_builtin_funcs == "STDDEV_POP"
        or numeric_agg_builtin_funcs == "VAR_POP"
    ):
        return

    # bitwise aggregate function only valid on integers
    if numeric_agg_builtin_funcs in {"BIT_XOR", "BIT_OR", "BIT_AND"}:
        if not np.issubdtype(bodosql_numeric_types["table1"]["A"].dtype, np.integer):
            return

    query = f"select {numeric_agg_builtin_funcs}(B), {numeric_agg_builtin_funcs}(C) from table1 group by A"

    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, check_names=False
    )


def test_agg_numeric_larger_group(
    grouped_dfs, numeric_agg_builtin_funcs, spark_info, memory_leak_check
):
    """test aggregation calls in queries on DataFrames with a larger data in each group."""
    # Skipping VAR_POP and STDDEV_POP with groupby due to [BE-910]
    if (
        numeric_agg_builtin_funcs == "STDDEV_POP"
        or numeric_agg_builtin_funcs == "VAR_POP"
    ):
        return

    # bitwise aggregate function only valid on integers
    if numeric_agg_builtin_funcs in {"BIT_XOR", "BIT_OR", "BIT_AND"}:
        if not np.issubdtype(bodosql_numeric_types["table1"]["A"].dtype, np.integer):
            return

    query = f"select {numeric_agg_builtin_funcs}(B), {numeric_agg_builtin_funcs}(C) from table1 group by A"

    check_query(query, grouped_dfs, spark_info, check_dtype=False, check_names=False)


@pytest.mark.slow
def test_aliasing_agg_numeric(
    bodosql_numeric_types, numeric_agg_builtin_funcs, spark_info, memory_leak_check
):
    """test aliasing of aggregations in queries"""
    # Skipping VAR_POP and STDDEV_POP with groupby due to [BE-910]
    if (
        numeric_agg_builtin_funcs == "STDDEV_POP"
        or numeric_agg_builtin_funcs == "VAR_POP"
    ):
        return

    # bitwise aggregate function only valid on integers
    if numeric_agg_builtin_funcs in {"BIT_XOR", "BIT_OR", "BIT_AND"}:
        if not np.issubdtype(bodosql_numeric_types["table1"]["A"].dtype, np.integer):
            return

    query = f"select {numeric_agg_builtin_funcs}(B) as testCol from table1 group by A"

    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, check_names=False
    )


def test_count_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """test various count queries on numeric data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1 group by A",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1 group by A",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_count_nullable_numeric(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """test various count queries on nullable numeric data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1 group by A",
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1 group by A",
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_count_datetime(bodosql_datetime_types, spark_info, memory_leak_check):
    """test various count queries on Timestamp data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1 group by A",
        bodosql_datetime_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1 group by A",
        bodosql_datetime_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_count_interval(bodosql_interval_types, spark_info, memory_leak_check):
    """test various count queries on Timedelta data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1 group by A",
        bodosql_interval_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1 group by A",
        bodosql_interval_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_count_string(bodosql_string_types, spark_info, memory_leak_check):
    """test various count queries on string data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1 group by A",
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1 group by A",
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_count_binary(bodosql_binary_types, spark_info, memory_leak_check):
    """test various count queries on binary data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1 group by A",
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1 group by A",
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_count_boolean(bodosql_boolean_types, spark_info, memory_leak_check):
    """test various count queries on boolean data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1 group by A",
        bodosql_boolean_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1 group by A",
        bodosql_boolean_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_count_numeric_alias(bodosql_numeric_types, spark_info, memory_leak_check):
    """test various count queries on numeric data with aliases."""
    check_query(
        "SELECT COUNT(Distinct B) as alias FROM table1 group by A",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) as alias FROM table1 group by A",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_having_repeat_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """test having clause in numeric queries"""
    check_query(
        "select sum(B) from table1 group by a having count(b) >1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_having_repeat_datetime(bodosql_datetime_types, spark_info, memory_leak_check):
    """test having clause in datetime queries"""
    check_query(
        f"select count(B) from table1 group by a having count(b) > 2",
        bodosql_datetime_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_having_repeat_interval(bodosql_interval_types, spark_info, memory_leak_check):
    """test having clause in datetime queries"""
    check_query(
        f"select count(B) from table1 group by a having count(b) > 2",
        bodosql_interval_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_agg_repeat_col(bodosql_numeric_types, spark_info, memory_leak_check):
    """test aggregations repeating the same column"""
    check_query(
        "select max(A), min(A), avg(A) from table1 group by B",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_groupby_bool(bodosql_boolean_types, spark_info, memory_leak_check):
    """
    Simple test to ensure that groupby and max are working on boolean types
    """
    query = """
        SELECT
            max(B)
        FROM
            table1
        GROUP BY
            A
        """
    check_query(
        query, bodosql_boolean_types, spark_info, check_names=False, check_dtype=False
    )


def test_groupby_string(bodosql_string_types, spark_info, memory_leak_check):
    """
    Simple test to ensure that groupby and max are working on string types
    """
    query = """
        SELECT
            max(B)
        FROM
            table1
        GROUP BY
            A
        """
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_names=False,
    )


@pytest.mark.slow
def test_groupby_numeric_scalars(basic_df, spark_info, memory_leak_check):
    """
    tests to ensure that groupby and max are working with numeric scalars
    """
    query = """
        SELECT
            max(B), 1 as E, 2 as F
        FROM
            table1
        GROUP BY
            A
        """
    check_query(query, basic_df, spark_info, check_names=False)


def test_groupby_datetime_types(bodosql_datetime_types, spark_info, memory_leak_check):
    """
    Simple test to ensure that groupby and max are working on datetime types
    """
    query = """
        SELECT
            max(B)
        FROM
            table1
        GROUP BY
            A
        """
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_names=False,
    )


def test_groupby_interval_types(bodosql_interval_types, spark_info, memory_leak_check):
    """
    Simple test to ensure that groupby and max are working on interval types
    """
    query = """
        SELECT
            max(B) as output
        FROM
            table1
        GROUP BY
            A
        """
    check_query(
        query,
        bodosql_interval_types,
        spark_info,
        convert_columns_timedelta=["output"],
    )


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("SELECT COUNT_IF(A) FROM table1 GROUP BY B", id="groupby_string"),
        pytest.param(
            "SELECT COUNT_IF(A) FROM table1 GROUP BY C",
            id="groupby_int",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT COUNT_IF(A) FROM table1 GROUP BY A",
            id="groupby_bool",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT COUNT_IF(A) FROM table1 GROUP BY B, C", id="groupby_stringInt"
        ),
        pytest.param(
            "SELECT COUNT_IF(A) FROM table1 GROUP BY A, B",
            id="groupby_stringBool",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT COUNT_IF(C > 3), COUNT_IF(C IS NULL OR C < 2) FROM table1 GROUP BY B",
            id="groupby_string_with_condition",
        ),
    ],
)
def test_count_if(query, spark_info, memory_leak_check):
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": pd.Series(
                    [True, False, True, None, True, None, True, False] * 5,
                    dtype=pd.BooleanDtype(),
                ),
                "B": pd.Series(list("AABAABCBAC") * 4),
                "C": pd.Series((list(range(7)) + [None]) * 5, dtype=pd.Int32Dtype()),
            }
        )
    }

    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_having_numeric(
    bodosql_numeric_types,
    comparison_ops,
    spark_info,
    # seems to be leaking memory sporatically, see [BS-534]
    # memory_leak_check
):
    """
    Tests having with a constant and groupby.
    """
    query = f"""
        SELECT
           MAX(A)
        FROM
            table1
        Group By
            C
        HAVING
            max(B) {comparison_ops} 1
        """
    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_having_boolean_agg_cond(bodosql_boolean_types, spark_info, memory_leak_check):
    """
    Tests groupby + having with aggregation in the condition
    """
    query = f"""
        SELECT
           MAX(A)
        FROM
            table1
        GROUP BY
            C
        HAVING
            max(B) <> True
        """
    check_query(
        query,
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_having_boolean_groupby_cond(
    bodosql_boolean_types, spark_info, memory_leak_check
):
    """
    Tests groupby + having using the groupby column in the having condtion
    """
    query = f"""
        SELECT
           MAX(A)
        FROM
            table1
        GROUP BY
            C
        HAVING
            C <> True
        """
    check_query(
        query,
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_repeat_columns(basic_df, spark_info, memory_leak_check):
    """
    Tests that a column that won't produce a conflicting name
    even if it performs the same operation.
    """
    query = "Select sum(A), sum(A) as alias from table1 group by B"
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_no_rename(basic_df, spark_info, memory_leak_check):
    """
    Tests that a columns with legal Python identifiers aren't renamed
    in simple queries (no intermediate names).
    """
    query = "Select sum(A) as col1, max(A) as col2 from table1 group by B"
    result = check_query(
        query, basic_df, spark_info, check_dtype=False, return_codegen=True
    )
    pandas_code = result["pandas_code"]
    assert "rename" not in pandas_code


@pytest.mark.slow
def test_rename(basic_df, spark_info, memory_leak_check):
    """
    Tests that a columns without a legal Python identifiers is renamed
    in a simple query (no intermediate names).
    """
    query = "Select sum(A) as `sum(A)`, max(A) as `max(A)` from table1 group by B"
    result = check_query(
        query, basic_df, spark_info, check_dtype=False, return_codegen=True
    )
    pandas_code = result["pandas_code"]
    assert "rename" in pandas_code


@pytest.fixture
def groupby_extension_table():
    """generates a simple fixture that ensures that all possible groupings produce an output"""

    vals = [1, 2, 3, 4, None]
    A = []
    B = []
    C = []

    A = vals * len(vals) * len(vals)
    for val in vals:
        B += [val] * len(vals)
        C += [val] * len(vals) * len(vals)

    B = B * len(vals)

    return {
        "table1": pd.DataFrame(
            {"A": A * 2, "B": B * 2, "C": C * 2, "D": np.arange(len(A) * 2)}
        )
    }


def test_cube(groupby_extension_table, spark_info, memory_leak_check):
    """
    Tests that bodosql can use snowflake's cube syntax in groupby.
    Note: Snowflake doesn't care about the spacing. CUBE (A, B, C)
    and CUBE(A, B, C) are both valid in snowflake and calcite.
    """

    bodosql_query = f"select A, B, C, SUM(D) from table1 GROUP BY CUBE (A, B, C)"
    # SparkSQL syntax varies slightly
    spark_query = f"select A, B, C, SUM(D) from table1 GROUP BY A, B, C WITH CUBE"

    check_query(
        bodosql_query,
        groupby_extension_table,
        spark_info,
        equivalent_spark_query=spark_query,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_rollup(groupby_extension_table, spark_info, memory_leak_check):
    """
    Tests that bodosql can use snowflake's rollup syntax in groupby.
    Note: Snowflake doesn't care about the spacing. ROLLUP (A, B, C)
    and ROLLUP(A, B, C) are both valid in snowflake and calcite.
    """

    bodosql_query = f"select A, B, C, SUM(D) from table1 GROUP BY ROLLUP(A, B, C)"
    # SparkSQL syntax varies slightly
    spark_query = f"select A, B, C, SUM(D) from table1 GROUP BY A, B, C WITH ROLLUP"

    check_query(
        bodosql_query,
        groupby_extension_table,
        spark_info,
        equivalent_spark_query=spark_query,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_grouping_sets(groupby_extension_table, spark_info, memory_leak_check):
    """
    Tests that bodosql can use snowflake's grouping sets syntax in groupby.
    GROUPING SETS ((A, B), (C, B), (A), ()) and GROUPING SETS((A, B), (C, B), (A), ())
    are both valid.
    """

    # Note that duplicate grouping sets do have an effect on the output
    query = f"select A, B, C, SUM(D) from table1 GROUP BY GROUPING SETS((A, B), (), (), (A, B), (C, B), (A))"

    check_query(
        query,
        groupby_extension_table,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_no_group_or_agg(groupby_extension_table, spark_info, memory_leak_check):
    """Tests the case in which we have at least one empty group with no aggregation"""

    bodosql_query = f"select A, B from table1 GROUP BY rollup(A, B)"
    # SparkSQL syntax varies slightly
    spark_query = f"select A, B from table1 GROUP BY A, B WITH ROLLUP"

    check_query(
        bodosql_query,
        groupby_extension_table,
        spark_info,
        equivalent_spark_query=spark_query,
        check_dtype=False,
        check_names=False,
        sort_output=True,
    )


@pytest.mark.slow
def test_nested_grouping_clauses(
    groupby_extension_table, spark_info, memory_leak_check
):
    """
    Tests having nested grouping clauses in groupby. This is not valid SNOWFLAKE
    Syntax, but calcite supports it, so we might as well.
    """

    bodosql_query = (
        f"select * from table1 GROUP BY GROUPING SETS ((), rollup(A, D), cube(C, B))"
    )
    # SparkSQL doesn't allow nested grouping sets
    # These grouping sets are the expanded version of the above
    # Note the duplicate grouping sets ARE required for correct behavior
    spark_query = f"select * from table1 GROUP BY GROUPING SETS ((), (A, D), (A), (), (), (C), (B), (C, B))"

    check_query(
        bodosql_query,
        groupby_extension_table,
        spark_info,
        equivalent_spark_query=spark_query,
        check_dtype=False,
        check_names=False,
        sort_output=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (["A"], ["B"]),
            id="int32_groupby_string",
        ),
        pytest.param((["B"], ["A"]), id="string_groupby_int32", marks=pytest.mark.slow),
        pytest.param((["C"], ["B"]), id="float_groupby_string", marks=pytest.mark.slow),
        pytest.param(
            (["C"], ["A", "B"]), id="float_groupby_int32_string", marks=pytest.mark.slow
        ),
        pytest.param(
            (["A", "B", "C"], ["B"]),
            id="all_groupby_string",
        ),
        pytest.param(
            (["A", "B", "C"], ["A", "B", "C"]),
            id="all_groupby_all",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_any_value(args, spark_info, memory_leak_check):
    """Tests ANY_VALUE, which is normally nondeterministic but has been
    implemented in a way that is reproducible (by always returning the first
    value)"""
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": pd.Series([1, 2, 3, 4, None] * 5, dtype=pd.Int32Dtype()),
                "B": pd.Series(["A", "B", None, "C", "D"] * 5),
                "C": pd.Series([1.0, None, None, 13.0, -1.5] * 5),
            }
        )
    }

    val_cols, group_cols = args
    query = (
        "SELECT "
        + ", ".join([f"ANY_VALUE({col})" for col in val_cols])
        + " FROM table1 GROUP BY "
        + ", ".join(group_cols)
    )
    equivalent_query = query.replace("ANY_VALUE", "FIRST")

    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=equivalent_query,
        # TODO[BE-3456]: enable dict-encoded string test when segfault is fixed
        use_dict_encoded_strings=False,
    )
