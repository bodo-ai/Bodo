# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL aggregation operations with groupby on BodoSQL
"""
import string

import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query, get_equivalent_spark_agg_query


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

    # bitwise aggregate function only valid on integers
    if numeric_agg_builtin_funcs in {"BIT_XOR", "BIT_OR", "BIT_AND"}:
        if not np.issubdtype(bodosql_numeric_types["table1"]["A"].dtype, np.integer):
            return

    query = f"select {numeric_agg_builtin_funcs}(B), {numeric_agg_builtin_funcs}(C) from table1 group by A"

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        equivalent_spark_query=get_equivalent_spark_agg_query(query),
        check_dtype=False,
        check_names=False,
    )


def test_agg_numeric_larger_group(
    grouped_dfs, numeric_agg_builtin_funcs, spark_info, memory_leak_check
):
    """test aggregation calls in queries on DataFrames with a larger data in each group."""

    # bitwise aggregate function only valid on integers
    if numeric_agg_builtin_funcs in {"BIT_XOR", "BIT_OR", "BIT_AND"}:
        if not np.issubdtype(bodosql_numeric_types["table1"]["A"].dtype, np.integer):
            return

    query = f"select {numeric_agg_builtin_funcs}(B), {numeric_agg_builtin_funcs}(C) from table1 group by A"

    check_query(
        query,
        grouped_dfs,
        spark_info,
        equivalent_spark_query=get_equivalent_spark_agg_query(query),
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_aliasing_agg_numeric(
    bodosql_numeric_types, numeric_agg_builtin_funcs, spark_info, memory_leak_check
):
    """test aliasing of aggregations in queries"""

    # bitwise aggregate function only valid on integers
    if numeric_agg_builtin_funcs in {"BIT_XOR", "BIT_OR", "BIT_AND"}:
        if not np.issubdtype(bodosql_numeric_types["table1"]["A"].dtype, np.integer):
            return

    query = f"select {numeric_agg_builtin_funcs}(B) as testCol from table1 group by A"

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        equivalent_spark_query=get_equivalent_spark_agg_query(query),
        check_dtype=False,
        check_names=False,
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


@pytest.mark.slow
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


@pytest.mark.slow
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
    memory_leak_check,
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
    query = 'Select sum(A) as "sum(A)", max(A) as "max(A)" from table1 group by B'
    spark_query = query.replace('"', "`")
    result = check_query(
        query,
        basic_df,
        spark_info,
        equivalent_spark_query=spark_query,
        check_dtype=False,
        return_codegen=True,
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

    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=get_equivalent_spark_agg_query(query),
        # TODO[BE-3456]: enable dict-encoded string test when segfault is fixed
        use_dict_encoded_strings=False,
    )


@pytest.mark.parametrize(
    "query, res",
    [
        pytest.param(
            "SELECT boolor_agg(I) FROM table1 GROUP BY G",
            [None, False, True, True, True, True, True],
            id="boolor_agg-int",
        ),
        pytest.param(
            "SELECT boolor_agg(B) FROM table1 GROUP BY G",
            [None, False, True, True, True, True, True],
            id="boolor_agg-bool",
        ),
        pytest.param(
            "SELECT boolor_agg(F) FROM table1 GROUP BY G",
            [None, False, True, True, True, True, True],
            id="boolor_agg-float",
        ),
        pytest.param(
            "SELECT booland_agg(I) FROM table1 GROUP BY G",
            [None, False, False, False, True, True, True],
            id="booland_agg-int",
        ),
        pytest.param(
            "SELECT booland_agg(B) FROM table1 GROUP BY G",
            [None, False, False, False, True, True, True],
            id="booland_agg-bool",
        ),
        pytest.param(
            "SELECT booland_agg(F) FROM table1 GROUP BY G",
            [None, False, False, False, True, True, True],
            id="booloand_agg-float",
        ),
        pytest.param(
            "SELECT boolxor_agg(I) FROM table1 GROUP BY G",
            [None, False, True, True, False, True, False],
            id="boolxor_agg-int",
        ),
        pytest.param(
            "SELECT boolxor_agg(B) FROM table1 GROUP BY G",
            [None, False, True, True, False, True, False],
            id="boolxor_agg-bool",
        ),
        pytest.param(
            "SELECT boolxor_agg(F) FROM table1 GROUP BY G",
            [None, False, True, True, False, True, False],
            id="boolxor_agg-float",
        ),
    ],
)
def test_boolor_booland_boolxor_agg(query, res, memory_leak_check):
    """Tests boolor_agg, booland_agg and boolxor_agg. These is done separately
    from existing aggregation tests, as we need specific inputs to stress this function
    """

    ctx = {
        "table1": pd.DataFrame(
            {
                "I": pd.Series(
                    [
                        # Group 0: all NULL
                        None,
                        None,
                        None,
                        None,
                        # Group 1: all zero
                        0,
                        0,
                        0,
                        0,
                        # Group 2: one nonzero
                        -1,
                        0,
                        0,
                        0,
                        # Group 3: two null, one zero, one nonzero
                        None,
                        1,
                        None,
                        0,
                        # Group 4: two null, two nonzero
                        1234,
                        None,
                        -1232,
                        None,
                        # Group 5: three null, one nonzero
                        2048,
                        None,
                        None,
                        None,
                        # Group 6: all nonzero
                        3,
                        1,
                        4,
                        1,
                    ],
                    dtype=pd.Int64Dtype(),
                ),
                "B": pd.Series(
                    [
                        # Group 0: all NULL
                        None,
                        None,
                        None,
                        None,
                        # Group 1: all false
                        False,
                        False,
                        False,
                        False,
                        # Group 2: one true
                        True,
                        False,
                        False,
                        False,
                        # Group 3: two null, one true
                        None,
                        True,
                        None,
                        False,
                        # Group 4: two null, two true
                        True,
                        None,
                        True,
                        None,
                        # Group 5: three null, one true
                        None,
                        True,
                        None,
                        None,
                        # Group 6: all true
                        True,
                        True,
                        True,
                        True,
                    ],
                    dtype=pd.BooleanDtype(),
                ),
                "F": pd.array(
                    [
                        # Group 0: all NULL
                        None,
                        None,
                        None,
                        None,
                        # Group 1: all zero
                        0,
                        0,
                        0,
                        0,
                        # Group 2: one nonzero
                        1.0,
                        0,
                        0,
                        0,
                        # Group 3: two null, one zero, one nonzero
                        None,
                        36.0,
                        None,
                        0.0,
                        # Group 4: two null, two nonzero
                        2.0,
                        None,
                        None,
                        0.718281828,
                        # Group 5: three null, one nonzero
                        None,
                        None,
                        3.1415926,
                        None,
                        # Group 6: all nonzero
                        10.0,
                        1.01,
                        -0.1,
                        1.11,
                    ]
                ),
                "G": pd.Series(list(range(7))).repeat(4).values,
            }
        )
    }
    expected_output = pd.DataFrame({0: pd.Series(res, dtype="boolean")})

    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_booland_agg_having(memory_leak_check):
    """Test having with booland_agg aggregation in the condition"""
    query = (
        "SELECT G, boolor_agg(B) FROM table1 GROUP BY G HAVING booland_agg(B = True)"
    )
    expected_output = pd.DataFrame({"0: ": [4, 5, 6], "1: ": [True, True, True]})
    ctx = {
        "table1": pd.DataFrame(
            {
                "B": pd.Series(
                    [
                        # Group 0: all NULL
                        None,
                        None,
                        None,
                        None,
                        # Group 1: all false
                        False,
                        False,
                        False,
                        False,
                        # Group 2: one true
                        True,
                        False,
                        False,
                        False,
                        # Group 3: two null, one true
                        None,
                        True,
                        None,
                        False,
                        # Group 4: two null, two true
                        True,
                        None,
                        True,
                        None,
                        # Group 5: three null, one true
                        None,
                        True,
                        None,
                        None,
                        # Group 6: all true
                        True,
                        True,
                        True,
                        True,
                    ],
                    dtype=pd.BooleanDtype(),
                ),
                "G": pd.Series(list(range(7))).repeat(4).values,
            }
        )
    }
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_boolor_agg_output_type(memory_leak_check):
    """Test boolor_agg to verify the output type is boolean"""
    query = """WITH TEMP AS(SELECT boolor_agg(A) as agg_A, B FROM table1 GROUP BY B)
        SELECT COUNT (DISTINCT CASE WHEN agg_A then B end) as totals
        from TEMP"""
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": pd.Series(
                    [
                        None,
                        None,
                        None,
                        None,
                        0,
                        0,
                        0,
                        0,
                        -1,
                        0,
                        0,
                        0,
                        None,
                        1,
                        None,
                        0,
                        1234,
                        None,
                        -1232,
                        None,
                    ],
                    dtype=pd.Int64Dtype,
                ),
                "B": [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [None] * 4,
            }
        )
    }
    expected_output = pd.DataFrame(
        {
            "totals": pd.Series([2], dtype="Int64"),
        }
    )
    check_query(
        query,
        ctx,
        None,  # Spark info
        check_dtype=False,
        expected_output=expected_output,
        is_out_distributed=False,
    )


@pytest.mark.tz_aware
def test_max_min_tz_aware(memory_leak_check):
    """
    Test max and min on a tz-aware timestamp column
    """
    S = pd.Series(
        list(pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz="Poland"))
        + [None] * 5
    )
    df = pd.DataFrame({"A": S, "id": ["a", "b", "c", "a", "d"] * 7})
    ctx = {"table1": df}
    py_output = pd.DataFrame(
        {"output1": df.groupby("id").max()["A"], "output2": df.groupby("id").min()["A"]}
    )
    query = "Select max(A) as output1, min(A) as output2 from table1 group by id"
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
    )


@pytest.mark.tz_aware
def test_count_tz_aware(memory_leak_check):
    """
    Test count and count(*) on a tz-aware timestamp column
    """
    S = pd.Series(
        list(pd.date_range(start="1/1/2022", freq="16D5H", periods=30, tz="Poland"))
        + [None] * 5
    )
    df = pd.DataFrame({"A": S, "id": ["a", "b", "c", "a", "d"] * 7})
    ctx = {"table1": df}
    py_output = pd.DataFrame(
        {"output1": df.groupby("id").count()["A"], "output2": df.groupby("id").size()}
    )
    query = "Select count(A) as output1, Count(*) as output2 from table1 group by id"
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
    )


@pytest.mark.tz_aware
@pytest.mark.slow
def test_any_value_tz_aware(memory_leak_check):
    """
    Test any_value on a tz-aware timestamp column
    """
    df = pd.DataFrame(
        {
            # Any Value is not defined so keep everything in each group the same.
            "A": [
                pd.Timestamp("2022/1/1", tz="Poland"),
                pd.Timestamp("2022/1/2", tz="Poland"),
                pd.Timestamp("2022/1/3", tz="Poland"),
                pd.Timestamp("2022/1/1", tz="Poland"),
                pd.Timestamp("2022/1/4", tz="Poland"),
            ]
            * 7,
            "id": ["a", "b", "c", "a", "d"] * 7,
        }
    )
    ctx = {"table1": df}
    py_output = pd.DataFrame({"output1": df.groupby("id").head(1)["A"]})
    query = "Select ANY_VALUE(A) as output1 from table1 group by id"
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
    )


@pytest.mark.tz_aware
def test_tz_aware_key(memory_leak_check):
    """
    Test tz_aware values as a key to groupby.
    """
    df = pd.DataFrame(
        {
            "A": [
                pd.Timestamp("2022/1/1", tz="Poland"),
                pd.Timestamp("2022/1/2", tz="Poland"),
                pd.Timestamp("2022/1/3", tz="Poland"),
                pd.Timestamp("2022/1/1", tz="Poland"),
                pd.Timestamp("2022/1/4", tz="Poland"),
            ]
            * 7,
            "val": np.arange(35),
        }
    )
    ctx = {"table1": df}
    py_output = pd.DataFrame({"output1": df.groupby("A").sum()["val"]})
    query = "Select SUM(val) as output1 from table1 group by A"
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
    )


@pytest.mark.tz_aware
@pytest.mark.slow
def test_tz_aware_having(memory_leak_check):
    """
    Test having with tz-aware values.
    """
    df = pd.DataFrame(
        {
            "A": [
                pd.Timestamp("2022/1/1", tz="Poland"),
                pd.Timestamp("2022/1/2", tz="Poland"),
                pd.Timestamp("2022/1/3", tz="Poland"),
                pd.Timestamp("2022/1/1", tz="Poland"),
                pd.Timestamp("2022/1/4", tz="Poland"),
            ]
            * 7,
            "B": [
                pd.Timestamp("2021/1/12", tz="Poland"),
                pd.Timestamp("2022/2/4", tz="Poland"),
                pd.Timestamp("2021/1/4", tz="Poland"),
                None,
                pd.Timestamp("2022/1/1", tz="Poland"),
                pd.Timestamp("2027/1/1", tz="Poland"),
                None,
            ]
            * 5,
            "C": [1, 2, 3] * 10 + [1, 2, 3, 4, 5],
        }
    )
    ctx = {"table1": df}

    def groupby_func(df):
        if df.A.max() > df.B.min():
            return df.A.min()
        else:
            return None

    py_output = df.groupby("C", as_index=False).apply(groupby_func)
    # Rename the columns
    py_output.columns = ["C", "output1"]
    # Remove the rows having would remove
    py_output = py_output[py_output.output1.notna()]
    query = "Select C, MIN(A) as output1 from table1 group by C HAVING max(A) > min(B)"
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
    )


def test_all_nulls(memory_leak_check):
    """
    Test the case where all values in a group are null.
    """
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5] * 6,
            # keep 1 value non-null to enable types in spark
            "B": pd.Series([1] + [None] * 29, dtype="Int64"),
            "C": pd.Series([False] + [None] * 29, dtype="boolean"),
        }
    )
    py_output = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "out1": pd.Series([1, None, None, None, None], dtype="Int64"),
            "out2": pd.Series([1, None, None, None, None], dtype="Int64"),
            "out3": pd.Series([1, None, None, None, None], dtype="Int64"),
            "out4": pd.Series([1, 0, 0, 0, 0], dtype="int64"),
            "out5": pd.Series([1.0, None, None, None, None], dtype="Float64"),
            "out6": pd.Series([None, None, None, None, None], dtype="Float64"),
            "out7": pd.Series([None, None, None, None, None], dtype="Float64"),
            "out8": pd.Series([False, None, None, None, None], dtype="boolean"),
        }
    )
    ctx = {"table1": df}
    query = """
        Select A,
        SUM(B) as out1,
        MAX(B) as out2,
        MIN(B) as out3,
        COUNT(B) as out4,
        AVG(B) as out5,
        STDDEV(B) as out6,
        VARIANCE(B) as out7,
        BOOLOR_AGG(C) AS out8
    from table1 group by A
    """
    check_query(query, ctx, None, check_dtype=False, expected_output=py_output)


@pytest.mark.parametrize(
    "agg_cols",
    [
        pytest.param("BD", id="fast_tests_a"),
        pytest.param("GK", id="fast_tests_b"),
        pytest.param("CEF", id="slow_tests_a", marks=pytest.mark.slow),
        pytest.param("HIJ", id="slow_tests_b", marks=pytest.mark.slow),
    ],
)
def test_kurtosis_skew(agg_cols, spark_info, memory_leak_check):
    """Tests the Kurtosis and Skew functions"""
    query = (
        "SELECT "
        + ", ".join(f"Skew({col}), Kurtosis({col})" for col in agg_cols)
        + " FROM table1 GROUP BY A"
    )
    # Datasets designed to exhibit different distributions, thus producing myriad
    # cases of kurtosis and skew calculations
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": pd.Series(
                    [int(np.log2(i**2 + 10)) for i in range(100)],
                    dtype=pd.Int32Dtype(),
                ),
                "B": pd.Series([float(i) for i in range(100)]),
                "C": pd.Series([i**2 for i in range(100)], dtype=pd.Int32Dtype()),
                "D": pd.Series(
                    [None if i % 2 == 0 else i for i in range(100)],
                    dtype=pd.Int32Dtype(),
                ),
                "E": pd.Series(
                    [None if i % 3 == 0 else float(i**2) for i in range(100)]
                ),
                "F": pd.Series([float((i**3) % 100) for i in range(100)]),
                "G": pd.Series([2.718281828 for i in range(100)]),
                "H": pd.Series([(i / 100) ** 0.5 for i in range(100)]),
                "I": pd.Series(
                    [np.arctanh(np.pi * (i - 49.5) / 160.5) for i in range(100)]
                ),
                "J": pd.Series([np.cbrt(i) for i in range(-49, 50)]),
                "K": pd.Series([max(0, (i - 40) // 15) for i in range(100)]),
            }
        )
    }

    def kurt_skew_refsol(cols):
        result = pd.DataFrame({"A0": ctx["table1"]["A"].drop_duplicates()})
        i = 0
        for col in cols:
            result[f"A{i}"] = (
                ctx["table1"]
                .groupby("A")
                .agg(res=pd.NamedAgg(col, aggfunc=pd.Series.skew))["res"]
                .values
            )
            i += 1
            result[f"A{i}"] = (
                ctx["table1"]
                .groupby("A")
                .agg(res=pd.NamedAgg(col, aggfunc=pd.Series.kurtosis))["res"]
                .values
            )
            i += 1
        return result

    answer = kurt_skew_refsol(agg_cols)

    check_query(
        query,
        ctx,
        None,
        expected_output=answer,
        check_names=False,
        check_dtype=False,
    )


@pytest.fixture()
def listagg_data():
    """When doing listagg without any grouping, the order is completely random.
    Therefore, to avoid situations, we include several columns for which the
    value is always the same per group.
    """
    return {
        "table1": pd.DataFrame(
            {
                "key_col": [1] * 6 + [2] * 6,
                "group_constant_str_col": ["a"] * 6 + ["œ"] * 6,
                "group_constant_str_col2": ["į"] * 6 + ["ë"] * 6,
                "non_constant_str_col": list(string.ascii_uppercase[:6]) * 2,
                "order_col_1": [None, 1, None, 2, None, 3] * 2,
                "order_col_2": [1, None, 2, None, 3, None] * 2,
            }
        )
    }


def test_listagg_no_withing_group_no_sep(listagg_data, spark_info, memory_leak_check):
    """Simple listagg test without sorting."""

    spark_equiv_query = """SELECT array_join(collect_list(table1.group_constant_str_col), '') FROM table1 group by key_col"""
    check_query(
        "SELECT listagg(group_constant_str_col) FROM table1 group by key_col",
        listagg_data,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_equiv_query,
    )


@pytest.mark.slow
def test_listagg_no_withing_group_with_sep(listagg_data, spark_info, memory_leak_check):
    spark_equiv_query = """SELECT array_join(collect_list(table1.group_constant_str_col), ', ') FROM table1 group by key_col"""
    check_query(
        "SELECT listagg(group_constant_str_col, ', ') FROM table1 group by key_col",
        listagg_data,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_equiv_query,
    )


@pytest.mark.slow
def test_listagg_no_within_group_with_other_aggregates(
    listagg_data, spark_info, memory_leak_check
):
    spark_equiv_query = """SELECT MAX(group_constant_str_col), key_col, array_join(collect_list(table1.group_constant_str_col), ''), MAX(group_constant_str_col2), array_join(collect_list(table1.group_constant_str_col2), 'å´îøü') FROM table1 group by key_col"""
    check_query(
        "SELECT MAX(group_constant_str_col), key_col, listagg(group_constant_str_col), MAX(group_constant_str_col2), listagg(group_constant_str_col2, 'å´îøü') FROM table1 group by key_col",
        listagg_data,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_equiv_query,
    )


def test_listagg_within_group_sorting_simple(listagg_data, memory_leak_check):
    """tests a simple withing group sorting with only one call"""
    expected = pd.DataFrame(
        {
            "agg_output_col": ["F, D, B, A, C, E"] * 2,
        }
    )

    check_query(
        "SELECT listagg(non_constant_str_col, ', ') WITHIN GROUP (ORDER BY order_col_1 DESC NULLS LAST, order_col_2 ASC NULLS FIRST) as agg_output_col FROM table1 group by key_col",
        listagg_data,
        None,  # Spark info
        check_names=False,
        check_dtype=False,
        expected_output=expected,
    )


@pytest.mark.slow
def test_listagg_withing_group_sorting_complex(listagg_data, memory_leak_check):
    query = """
SELECT
   LISTAGG(group_constant_str_col) as agg_output_col1,
   LISTAGG(group_constant_str_col2, ',') as agg_output_col2,
   LISTAGG(group_constant_str_col, '*') as agg_output_col3,
   LISTAGG(non_constant_str_col, ', ') WITHIN GROUP (ORDER BY order_col_1) as agg_output_col4,
   LISTAGG(non_constant_str_col) WITHIN GROUP (ORDER BY order_col_1 DESC NULLS LAST, order_col_2 ASC) as agg_output_col5,
   LISTAGG(non_constant_str_col) WITHIN GROUP (ORDER BY order_col_2 ASC NULLS FIRST, order_col_1 ASC NULLS FIRST) as agg_output_col6
FROM table1
GROUP BY key_col
"""
    expected = pd.DataFrame(
        {
            "agg_output_col1": ["aaaaaa", "œœœœœœ"],
            "agg_output_col2": ["į,į,į,į,į,į", "ë,ë,ë,ë,ë,ë"],
            "agg_output_col3": ["a*a*a*a*a*a", "œ*œ*œ*œ*œ*œ"],
            "agg_output_col4": ["B, D, F, A, C, E"] * 2,
            "agg_output_col5": ["FDBACE"] * 2,
            "agg_output_col6": ["BDFACE"] * 2,
        }
    )

    check_query(
        query,
        listagg_data,
        None,  # Spark info
        expected_output=expected,
    )
