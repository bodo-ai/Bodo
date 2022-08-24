# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL case queries on BodoSQL
"""
import pytest
from bodosql.tests.utils import check_query

from bodo.tests.utils import gen_nonascii_list


@pytest.fixture(
    params=[
        # TODO: Float literals (Spark outputs decimal type, not float)
        # (1.3, -3124.2, 0.0, 314.1),
        # Integer literals
        (341, -3, 0, 3443),
        # Boolean literals
        (True, False, False, True),
        # String literals
        ("'hello'", "'world'", "'goodbye'", "'spark'"),
        # TODO: Timestamp Literals (Cannot properly compare with Spark)
        # (
        #     "TIMESTAMP '1997-01-31 09:26:50.124'",
        #     "TIMESTAMP '2021-05-31 00:00:00.00'",
        #     "TIMESTAMP '2021-04-28 00:40:00.00'",
        #     "TIMESTAMP '2021-04-29'",
        # ),
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
            marks=pytest.mark.skip("[BE-957] Support Bytes.fromhex"),
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
    check_query(query, basic_df, spark_info, check_dtype=False)


@pytest.mark.slow
def test_case_literals_multiple_when(
    basic_df, case_literals, spark_info, memory_leak_check
):
    """
    Test a case statement with multiple whens and each possible literal return type.
    """

    query = f"Select B, Case WHEN A = 1 THEN {case_literals[0]} WHEN A = 2 THEN {case_literals[1]} WHEN B > 6 THEN {case_literals[2]} ELSE {case_literals[3]} END as CaseRes FROM table1"
    check_query(query, basic_df, spark_info, check_dtype=False)


@pytest.mark.slow
def test_case_literals_groupby(basic_df, case_literals, spark_info, memory_leak_check):
    """
    Test a case statement with each possible literal return type in a groupby.
    """
    query = f"Select B, Case WHEN A >= 2 THEN {case_literals[0]} ELSE {case_literals[1]} END as CaseRes FROM table1 Group By A, B"
    check_query(query, basic_df, spark_info, check_dtype=False)


@pytest.mark.slow
def test_case_literals_multiple_when_groupby(
    basic_df, case_literals, spark_info, memory_leak_check
):
    """
    Test a case statement with each possible literal return type in a groupby.
    """

    query = f"Select B, Case WHEN A = 1 THEN {case_literals[0]} WHEN A = 2 THEN {case_literals[1]} WHEN B > 6 THEN {case_literals[2]} ELSE {case_literals[3]} END as CaseRes FROM table1 Group By A, B"

    check_query(query, basic_df, spark_info, check_dtype=False)


@pytest.mark.skip
def test_case_literals_nonascii(basic_df, spark_info, memory_leak_check):
    """
    Test a case statement with non-ASCII literals.
    """
    case_literals = gen_nonascii_list(4)

    query = f"Select B, Case WHEN A = 1 THEN {case_literals[0]} WHEN A = 2 THEN {case_literals[1]} WHEN B > 6 THEN {case_literals[2]} ELSE {case_literals[3]} END as CaseRes FROM table1 Group By A, B"

    check_query(query, basic_df, spark_info, check_dtype=False)


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
    # See [BS-157], currently an issue with null valued boolean in case
    if isinstance(case_literals[0], bool):
        return
    query = f"Select Case WHEN A >= 2 THEN {case_literals[0]} WHEN A = 1 THEN {case_literals[1]} END FROM table1"
    check_query(query, basic_df, spark_info, check_dtype=False, check_names=False)


def test_case_no_else_clause_columns(basic_df, spark_info, memory_leak_check):
    """
    Test a case statement that doesn't have an else clause whoose values are columns
    """
    query = f"Select Case WHEN A >= 2 THEN A WHEN A < 0 THEN B END FROM table1"
    check_query(query, basic_df, spark_info, check_dtype=False, check_names=False)


def test_shortcircuit_and(zeros_df, spark_info, memory_leak_check):
    """tests that and when used in a case statement shortcircuits"""
    query = "Select Case WHEN B <> 0 AND (A / B > 0) THEN A ELSE B END FROM table1"
    check_query(query, zeros_df, spark_info, check_names=False, check_dtype=False)


def test_shortcircuit_or(zeros_df, spark_info, memory_leak_check):
    """tests that and when used in a case statement shortcircuits"""
    query = "Select Case WHEN B = 0 OR (A / B > 0) THEN A ELSE B END FROM table1"
    check_query(query, zeros_df, spark_info, check_names=False, check_dtype=False)
