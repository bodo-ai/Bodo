# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL case queries on BodoSQL
"""
import bodosql
import numba
import pandas as pd
import pytest
from bodosql.tests.utils import check_query
from numba.core import ir
from numba.core.ir_utils import find_callname, guard

import bodo
from bodo.tests.timezone_common import representative_tz  # noqa
from bodo.tests.utils import (
    ParforTestPipeline,
    SeriesOptTestPipeline,
    gen_nonascii_list,
)


@pytest.fixture(
    params=[
        # TODO: Float literals (Spark outputs decimal type, not float)
        pytest.param((1.3, -3124.2, 0.0, 314.1), id="float_literals"),
        # Integer literals
        (341, -3, 0, 3443),
        # Boolean literals
        (True, False, False, True),
        # String literals
        ("'hello'", "'world'", "'goodbye'", "'spark'"),
        # TODO: Timestamp Literals (Cannot properly compare with Spark)
        pytest.param(
            (
                "TIMESTAMP '1997-01-31 09:26:50.124'",
                "TIMESTAMP '2021-05-31 00:00:00.00'",
                "TIMESTAMP '2021-04-28 00:40:00.00'",
                "TIMESTAMP '2021-04-29'",
            ),
            id="timestamp_literals",
        ),
        # TODO: Interval Literals (Cannot convert to Pandas in Spark)
        # (
        #     "INTERVAL '1' year",
        #     "INTERVAL '1' day",
        #     "INTERVAL '-4' hours",
        #     "INTERVAL '3' months",
        # ),
        # Binary literals. These are hex byte values starting with X''
        pytest.param(
            (
                f"X'{b'hello'.hex()}'",
                f"X'{b'world'.hex()}'",
                f"X'{b'goodbye'.hex()}'",
                f"X'{b'spark'.hex()}'",
            ),
            marks=pytest.mark.skip("[BE-3304] Support Bytes literals"),
            id="binary_literals",
        ),
    ]
)
def case_literals(request):
    """
    Fixture of possible literal choices for generating
    case code. There are 4 values to support possible nested cases.
    """
    return request.param


@pytest.mark.slow
def test_case_agg(bodosql_string_types, spark_info, memory_leak_check):
    """
    Test using an aggregate function on the output of a case statement.
    """
    check_query(
        """
        SELECT SUM(CASE
          WHEN A = 'hello'
          THEN 1
          ELSE 0
         END) FROM table1""",
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_boolean_literal_case(basic_df, spark_info, memory_leak_check):
    """
    Tests the behavior of case when the possible results are boolean literals.
    """
    query1 = "Select B, Case WHEN A >= 2 THEN True ELSE True END as CaseRes FROM table1"
    query2 = (
        "Select B, Case WHEN A >= 2 THEN True ELSE False END as CaseRes FROM table1"
    )
    query3 = (
        "Select B, Case WHEN A >= 2 THEN False ELSE True END as CaseRes FROM table1"
    )
    query4 = (
        "Select B, Case WHEN A >= 2 THEN False ELSE False END as CaseRes FROM table1"
    )
    check_query(query1, basic_df, spark_info, check_dtype=False)
    check_query(query2, basic_df, spark_info, check_dtype=False)
    check_query(query3, basic_df, spark_info, check_dtype=False)
    check_query(query4, basic_df, spark_info, check_dtype=False)


def test_case_literals(basic_df, case_literals, spark_info, memory_leak_check):
    """
    Test a case statement with each possible literal return type.
    """
    query = f"Select B, Case WHEN A >= 2 THEN {case_literals[0]} ELSE {case_literals[1]} END as CaseRes FROM table1"
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        convert_columns_decimal=["CaseRes"],
    )


@pytest.mark.slow
def test_case_literals_multiple_when(
    basic_df, case_literals, spark_info, memory_leak_check
):
    """
    Test a case statement with multiple whens and each possible literal return type.
    """

    query = f"Select B, Case WHEN A = 1 THEN {case_literals[0]} WHEN A = 2 THEN {case_literals[1]} WHEN B > 6 THEN {case_literals[2]} ELSE {case_literals[3]} END as CaseRes FROM table1"
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        convert_columns_decimal=["CaseRes"],
    )


@pytest.mark.slow
def test_case_literals_groupby(basic_df, case_literals, spark_info, memory_leak_check):
    """
    Test a case statement with each possible literal return type in a groupby.
    """
    query = f"Select B, Case WHEN A >= 2 THEN {case_literals[0]} ELSE {case_literals[1]} END as CaseRes FROM table1 Group By A, B"
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        convert_columns_decimal=["CaseRes"],
    )


@pytest.mark.slow
def test_case_literals_multiple_when_groupby(
    basic_df, case_literals, spark_info, memory_leak_check
):
    """
    Test a case statement with each possible literal return type in a groupby.
    """

    query = f"Select B, Case WHEN A = 1 THEN {case_literals[0]} WHEN A = 2 THEN {case_literals[1]} WHEN B > 6 THEN {case_literals[2]} ELSE {case_literals[3]} END as CaseRes FROM table1 Group By A, B"

    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        convert_columns_decimal=["CaseRes"],
    )


def test_case_literals_nonascii(basic_df, spark_info, memory_leak_check):
    """
    Test a case statement with non-ASCII literals.
    """
    case_literals = gen_nonascii_list(4)

    query = f"Select B, Case WHEN A = 1 THEN '{case_literals[0]}' WHEN A = 2 THEN '{case_literals[1]}' WHEN B > 6 THEN '{case_literals[2]}' ELSE '{case_literals[3]}' END as CaseRes FROM table1 Group By A, B"

    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        convert_columns_decimal=["CaseRes"],
    )


@pytest.mark.slow
def test_case_agg_groupby(basic_df, spark_info, memory_leak_check):
    """
    Test a case statement with an aggregate function applied to each group.
    """
    query1 = f"Select Case WHEN A >= 2 THEN Sum(B) ELSE 0 END as CaseRes FROM table1 Group By A"
    query2 = f"Select Case WHEN A >= 2 THEN Count(B) ELSE 0 END as CaseRes FROM table1 Group By A"
    check_query(query1, basic_df, spark_info, check_dtype=False)
    check_query(query2, basic_df, spark_info, check_dtype=False)


@pytest.mark.slow
def test_case_no_else_clause_literals(
    basic_df, case_literals, spark_info, memory_leak_check
):
    """
    Test a case statement that doesn't have an else clause whoose values are scalars
    """
    query = f"Select Case WHEN A >= 2 THEN {case_literals[0]} WHEN A = 1 THEN {case_literals[1]} END as CaseRes FROM table1"
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        convert_columns_decimal=["CaseRes"],
    )


def test_case_no_else_clause_columns(basic_df, spark_info, memory_leak_check):
    """
    Test a case statement that doesn't have an else clause whoose values are columns
    """
    query = f"Select Case WHEN A >= 2 THEN A WHEN A < 0 THEN B END FROM table1"
    check_query(query, basic_df, spark_info, check_dtype=False, check_names=False)


def test_timestamp_to_datetime_opt(spark_info, memory_leak_check):
    """make sure pd.Timestamp()/pd.to_datetime() calls with constant string inputs are
    optimized out since they go to objmode and are very expensive"""

    query = "Select Case WHEN A > CAST('2022-10-31' AS DATE) THEN 1 END FROM table1"
    _check_timestamp_to_datetime_opt(spark_info, query, query)
    query = "Select Case WHEN A > STR_TO_DATE('2022-10-31', '%Y-%m-%d') THEN 1 END FROM table1"
    spark_query = "Select Case WHEN A > TO_DATE('2022-10-31', 'yyyy-MM-dd') THEN 1 END FROM table1"
    _check_timestamp_to_datetime_opt(spark_info, query, spark_query)


def test_tz_aware_case_null(representative_tz, memory_leak_check):
    """
    Tests a case statement using a column + NULL on tz-aware timestamp
    data.
    """
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                "2022/1/1", periods=30, freq="6D5H", tz=representative_tz
            ),
            "B": [True, False, True, True, False] * 6,
        }
    )
    query = "Select Case WHEN B THEN A END as output FROM table1"
    ctx = {"table1": df}
    expected_output = pd.DataFrame({"output": df["A"].copy()})
    expected_output[~df.B] = None
    check_query(query, ctx, None, expected_output=expected_output)


def _check_timestamp_to_datetime_opt(spark_info, query, spark_query):
    """make sure pd.Timestamp()/pd.to_datetime() with constant string input calls are
    optimized out for query
    """

    df = pd.DataFrame({"A": pd.date_range("2022-10-28", periods=10)})
    check_query(
        query,
        {"table1": df},
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )

    @bodo.jit(pipeline_class=ParforTestPipeline)
    def bodo_func(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(query)

    # Make sure there is no pd.Timestamp() in the IR
    bodo_func(df)
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]

    # get the CASE Parfor from the IR (should have only one Parfor)
    parfor = None
    for block in fir.blocks.values():
        for stmt in block.body:
            if isinstance(stmt, numba.parfors.parfor.Parfor):
                assert parfor is None, "only one parfor expected"
                parfor = stmt
    assert parfor is not None, "parfor not found"

    for block in parfor.loop_body.values():
        for stmt in block.body:
            if isinstance(stmt, ir.Assign):
                rhs = stmt.value
                if isinstance(rhs, ir.Expr) and rhs.op == "call":
                    fdef = guard(find_callname, fir, stmt.value)
                    assert fdef != (
                        "Timestamp",
                        "pandas",
                    ), "pd.Timestamp() found"
                    assert fdef != (
                        "sql_null_checking_pd_to_datetime_with_format",
                        "bodosql.libs.generated_lib",
                    ), "sql_null_checking_pd_to_datetime_with_format found"


def test_case_no_inlining(basic_df, spark_info, memory_leak_check):
    """
    Test a case that makes sure the no inlining path works as expected.
    """
    try:
        # Save the old threshold and set it to 1 so that the case statement
        # is not inlined.
        old_threshold = bodo.COMPLEX_CASE_THRESHOLD
        bodo.COMPLEX_CASE_THRESHOLD = 1
        query = f"Select B, Case WHEN A = 1 THEN 1 WHEN A = 2 THEN 2 WHEN B > 6 THEN 3 ELSE NULL END as CaseRes FROM table1"
        check_query(
            query,
            basic_df,
            spark_info,
            check_dtype=False,
        )
        # Add a check that the case statement is not inlined.
        @bodo.jit(pipeline_class=SeriesOptTestPipeline)
        def bodo_func(bc, query):
            return bc.sql(query)

        bodo_func(bodosql.BodoSQLContext(basic_df), query)
        fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
        # Inspect the IR for a bodosql_case_kernel call
        found = False
        for block in fir.blocks.values():
            for stmt in block.body:
                if isinstance(stmt, ir.Assign):
                    fdef = guard(find_callname, fir, stmt.value)
                    found = found or fdef == ("bodosql_case_kernel", "")
        assert found, "bodosql_case_kernel not found in IR, case statement was inlined"

    finally:
        # restore the old threshold
        bodo.COMPLEX_CASE_THRESHOLD = old_threshold
