# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL aggregation operations without groupby on BodoSQL
"""
import bodosql
import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query

import bodo
from bodo.tests.utils import (
    DistTestPipeline,
    check_func,
    count_array_OneDs,
    count_array_REPs,
    dist_IR_contains,
)


def test_agg_numeric(
    bodosql_numeric_types, numeric_agg_builtin_funcs, spark_info, memory_leak_check
):
    """test agg func calls in queries"""

    # bitwise aggregate function only valid on integers
    if numeric_agg_builtin_funcs in {"BIT_XOR", "BIT_OR", "BIT_AND"}:
        if not np.issubdtype(bodosql_numeric_types["table1"]["A"].dtype, np.integer):
            return

    query = f"select {numeric_agg_builtin_funcs}(B), {numeric_agg_builtin_funcs}(C) from table1"

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        is_out_distributed=False,
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

    query = f"select {numeric_agg_builtin_funcs}(B) as testCol from table1"

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_repeat_columns(basic_df, spark_info, memory_leak_check):
    """
    Tests that a column that won't produce a conflicting name
    even if it performs the same operation.
    """
    query = "Select sum(A), sum(A) as alias from table1"
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        is_out_distributed=False,
    )


def test_count_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """test various count queries on numeric data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_count_nullable_numeric(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """test various count queries on nullable numeric data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_count_datetime(bodosql_datetime_types, spark_info, memory_leak_check):
    """test various count queries on Timestamp data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_datetime_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_datetime_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_count_interval(bodosql_interval_types, spark_info, memory_leak_check):
    """test various count queries on Timedelta data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_interval_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_interval_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_count_boolean(bodosql_boolean_types, spark_info, memory_leak_check):
    """test various count queries on boolean data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_boolean_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_boolean_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


def test_count_string(bodosql_string_types, spark_info, memory_leak_check):
    """test various count queries on string data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


def test_count_binary(bodosql_binary_types, spark_info, memory_leak_check):
    """test various count queries on string data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_count_numeric_alias(bodosql_numeric_types, spark_info, memory_leak_check):
    """test various count queries on numeric data with aliases."""
    check_query(
        "SELECT COUNT(Distinct B) as alias FROM table1",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )
    check_query(
        "SELECT COUNT(*) as alias FROM table1",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.skip("[BS-81]")
def test_max_string(bodosql_string_types, spark_info, memory_leak_check):
    """
    Simple test to ensure that max is working on string types
    """
    query = """
        SELECT
            max(A)
        FROM
            table1
        """
    check_query(query, bodosql_string_types, spark_info, check_names=False)


def test_max_datetime_types(bodosql_datetime_types, spark_info, memory_leak_check):
    """
    Simple test to ensure that max is working on datetime types
    """
    query = """
        SELECT
            max(A)
        FROM
            table1
        """
    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_names=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_max_interval_types(bodosql_interval_types, spark_info, memory_leak_check):
    """
    Simple test to ensure that max is working on timedelta types
    """
    query = """
        SELECT
            max(A) as output
        FROM
            table1
        """
    check_query(
        query,
        bodosql_interval_types,
        spark_info,
        check_names=False,
        convert_columns_timedelta=["output"],
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_max_literal(basic_df, spark_info, memory_leak_check):
    """tests that max works on a scalar value"""
    # This query does not get optimized, by manual check
    query = "Select A, scalar_max from table1, (Select Max(1) as scalar_max)"

    check_query(
        query,
        basic_df,
        spark_info,
        # Max outputs a nullable output by default to handle empty Series values
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("SELECT COUNT_IF(A) FROM table1", id="bool_col"),
        pytest.param("SELECT COUNT_IF(B = 'B') FROM table1", id="string_match"),
        pytest.param("SELECT COUNT_IF(C % 2 = 1) FROM table1", id="int_cond"),
    ],
)
def test_count_if(query, spark_info, memory_leak_check):
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": pd.Series(
                    [True, False, None, True, True, None, False, True] * 5,
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
        is_out_distributed=False,
    )


def test_having(bodosql_numeric_types, comparison_ops, spark_info, memory_leak_check):
    """
    Tests having with a constant
    """
    query = f"""
        SELECT
           MAX(A)
        FROM
            table1
        HAVING
            max(B) {comparison_ops} 1
        """
    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        is_out_distributed=False,
    )


def test_max_bool(bodosql_boolean_types, spark_info, memory_leak_check):
    """
    Simple test to ensure that max is working on boolean types
    """
    query = """
        SELECT
            max(A)
        FROM
            table1
        """
    check_query(
        query,
        bodosql_boolean_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


def test_having_boolean(bodosql_boolean_types, spark_info, memory_leak_check):
    """
    Tests having with a constant
    """
    query = f"""
        SELECT
           MAX(A)
        FROM
            table1
        HAVING
            max(B) <> True
        """
    check_query(
        query,
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        is_out_distributed=False,
    )


def test_agg_replicated(memory_leak_check):
    """
    Tests that an aggregation query produces a
    replicated output.
    """

    def impl(filename):
        bc = bodosql.BodoSQLContext({"t1": bodosql.TablePath(filename, "parquet")})
        return bc.sql("select count(B) as cnt from t1")

    filename = "bodosql/tests/data/sample-parquet-data/no_index.pq"
    read_df = pd.read_parquet(filename)
    count = read_df.B.count()
    expected_output = pd.DataFrame({"cnt": count}, index=pd.Index([0]))
    check_func(impl, (filename,), py_output=expected_output, is_out_distributed=False)
    # Check that the function returns replicated data.
    bodo_func = bodo.jit(distributed_block={"A"}, pipeline_class=DistTestPipeline)(impl)
    bodo_func(filename)
    # This function needs to return at least two distributed outputs, the data column
    # and the index. This functions may contain more if there are aliases or other
    # intermediate variables.
    assert count_array_REPs() > 2, "Expected replicated return value"
    # The parquet data should still be loaded as 1D
    assert count_array_OneDs() > 1, "Expected distributed read from parquet"
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(
        f_ir, "dist_reduce"
    ), "Expected distributed reduction in the compute."


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("SELECT ANY_VALUE(A) FROM table1", id="int32"),
        pytest.param(
            "SELECT ANY_VALUE(B) FROM table1", id="string", marks=pytest.mark.slow
        ),
        pytest.param(
            "SELECT ANY_VALUE(C) FROM table1", id="float", marks=pytest.mark.slow
        ),
        pytest.param(
            "SELECT ANY_VALUE(A), ANY_VALUE(B), ANY_VALUE(C) FROM table1", id="all"
        ),
    ],
)
def test_any_value(query, spark_info, memory_leak_check):
    """Tests ANY_VALUE, which is normally nondeterministic but has been
    implemented in a way that is reproducible (by always returning the first
    value)"""
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": pd.Series(
                    [5, 3, 1, 10, 30, -1, 0, None, 3, 1, 5, 4, -1, None, 10] * 2,
                    dtype=pd.Int32Dtype(),
                ),
                "B": pd.Series(list("AABAABCBAABCDCB") * 2),
                "C": pd.Series(
                    [
                        (((i + 3) ** 2) % 50) / 10 if i % 5 != 2 else None
                        for i in range(30)
                    ]
                ),
            }
        )
    }

    equivalent_query = query.replace("ANY_VALUE", "FIRST")

    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=equivalent_query,
        is_out_distributed=False,
    )
