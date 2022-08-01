# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Test correctness of SQL string operation queries on BodoSQL
"""

import pytest

from bodosql.tests.string_ops_common import *  # noqa
from bodosql.tests.utils import check_query


def test_like(
    bodosql_string_types, regex_string, spark_info, like_expression, memory_leak_check
):
    """
    tests that like works for a variety of different possible regex strings
    """
    check_query(
        f"select A from table1 where A {like_expression} {regex_string}",
        bodosql_string_types,
        spark_info,
    )


@pytest.mark.slow
def test_like_scalar(
    bodosql_string_types, regex_string, spark_info, like_expression, memory_leak_check
):
    """
    tests that like works for a variety of different possible regex strings
    """
    check_query(
        f"select case when A {like_expression} {regex_string} then 1 else 0 end from table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
    )


@pytest.mark.slow
def test_like_with_logical_operators(
    bodosql_string_types, regex_string, spark_info, like_expression, memory_leak_check
):
    """
    test that like behaves well with logical operators
    """
    check_query(
        f"select A from table1 where A {like_expression} {regex_string} and B like {regex_string}",
        bodosql_string_types,
        spark_info,
    )
    check_query(
        f"select B from table1 where A {like_expression} {regex_string} or B like {regex_string}",
        bodosql_string_types,
        spark_info,
    )


def test_like_cols(
    basic_df, regex_string, spark_info, like_expression, memory_leak_check
):
    """tests that like is working in the column case"""
    check_query(
        f"select A from table1 where C {like_expression} {regex_string} or B {like_expression} {regex_string}",
        basic_df,
        spark_info,
        check_dtype=False,  # need this for case where the select retuns empty table
    )


@pytest.mark.slow
def test_like_constants(
    basic_df,
    regex_string,
    string_constants,
    spark_info,
    like_expression,
    memory_leak_check,
):
    """
    tests that like works on constant strings
    """
    query = f"select A from table1 where '{string_constants}' {like_expression} {regex_string}"
    check_query(query, basic_df, spark_info, check_dtype=False)


def test_nested_upper_lower(bodosql_string_types, spark_info):
    """
    Tests that lower/upper calls nest properly
    """
    check_query(
        f"select lower(upper(lower(upper(A)))) from table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
    )


def test_upper_lower_scalars(basic_df, string_constants, spark_info, memory_leak_check):
    """
    Tests that lower/upper calls work on scalar values
    """
    """
    "select A, upper('{string_constants}'), lower('{string_constants}') from table1" causes an issue, so for now,
    I'm just doing it as two seperate queries
    """
    query = f"select A, upper('{string_constants}') from table1"

    query2 = f"select A, lower('{string_constants}') from table1"

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
    )
    check_query(
        query2,
        basic_df,
        spark_info,
        check_names=False,
    )


@pytest.mark.slow
def test_upper_lower_scalars_nested(
    basic_df, string_constants, spark_info, memory_leak_check
):
    """
    Tests that nested lower/upper calls work on scalar values
    """
    query = f"select A, upper(lower(upper('{string_constants}'))) from table1"

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
    )


@pytest.mark.slow
def test_upper_lower_like_constants(
    basic_df,
    regex_string,
    string_constants,
    spark_info,
    like_expression,
    # memory_leak_check, Seems to be leaking memory sporatically, see [BS-534]
):
    """
    Tests that lower/upper works on string constants
    """
    check_query(
        f"select A from table1 where upper('{string_constants}') {like_expression} upper({regex_string})",
        basic_df,
        spark_info,
        check_dtype=False,
    )
    check_query(
        f"select A from table1 where lower('{string_constants}') {like_expression} upper({regex_string})",
        basic_df,
        spark_info,
        check_dtype=False,
    )
    check_query(
        f"select A from table1 where upper('{string_constants}') {like_expression} lower({regex_string})",
        basic_df,
        spark_info,
        check_dtype=False,
    )


@pytest.mark.slow
def test_pythonic_regex(
    bodosql_string_types,
    pythonic_regex,
    spark_info,
    like_expression,
    # TODO: re add memory_leak_check, see BS-534
):
    """
    checks that pythonic regex is working as inteded
    """
    result = check_query(
        f"select A from table1 where A {like_expression} '{pythonic_regex}'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_wildcardless_like(pandas_code)


@pytest.mark.slow
def test_all_percent(
    bodosql_string_types,
    spark_info,
    like_expression,
    # TODO: re add memory_leak_check, see BS-534
):
    """
    checks that a regex that is all %% is correct and properly optimized
    """
    result = check_query(
        f"select A from table1 where A {like_expression} '%%'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_start_and_end_percent_like(pandas_code)


@pytest.mark.slow
def test_all_percent_scalar(
    bodosql_string_types,
    spark_info,
    like_expression,
    # TODO: re add memory_leak_check, see BS-534
):
    """
    checks that a regex that is all %% is correct
    """
    check_query(
        f"select case when A {like_expression} '%%' then 1 else 0 end from table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
    )


@pytest.mark.slow
def test_leading_percent(
    bodosql_string_types,
    spark_info,
    like_expression,  # TODO: re add memory_leak_check, see BS-534
):
    """
    checks that a regex starting with % is correct and properly optimized
    """
    result = check_query(
        f"select A from table1 where A {like_expression} '%o'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_start_percent_like(pandas_code)

    result = check_query(
        f"select A from table1 where A {like_expression} '%.o'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_start_percent_like(pandas_code)


@pytest.mark.slow
def test_leading_percent_scalar(
    bodosql_string_types,
    spark_info,
    like_expression,  # TODO: re add memory_leak_check, see BS-534
):
    """
    checks that a regex starting with % is correct
    """
    check_query(
        f"select case when A {like_expression} '%.o' then 1 else 0 end from table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
    )


@pytest.mark.slow
def test_trailing_percent(
    bodosql_string_types,
    spark_info,
    like_expression,
    # TODO: re add memory_leak_check, see BS-534
):
    """
    checks that a regex ending with % is correct and properly optimized
    """
    result = check_query(
        f"select A from table1 where A {like_expression} 'h%'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_end_percent_like(pandas_code)

    result = check_query(
        f"select A from table1 where A {like_expression} 'h.%'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_end_percent_like(pandas_code)


@pytest.mark.slow
def test_trailing_percent_scalar(
    bodosql_string_types,
    spark_info,
    like_expression
    # TODO: re add memory_leak_check, see BS-534
):
    """
    checks that a regex ending with % is correct
    """
    check_query(
        f"select case when A {like_expression} '%.o' then 1 else 0 end from table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
    )


@pytest.mark.slow
def test_both_percent(
    bodosql_string_types,
    spark_info,
    like_expression,
    # memory_leak_check Seems to be failing memory leak check intermitently, see BS-534
):
    """
    checks that a regex starting and ending with % is correct and properly optimized
    """
    result = check_query(
        f"select A from table1 where A {like_expression} '%e%'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_start_and_end_percent_like(pandas_code)

    result = check_query(
        f"select A from table1 where A {like_expression} '%e.%'",
        bodosql_string_types,
        spark_info,
        return_codegen=True,
    )
    pandas_code = result["pandas_code"]
    check_start_and_end_percent_like(pandas_code)


@pytest.mark.slow
def test_both_percent_scalar(
    bodosql_string_types,
    spark_info,
    like_expression,
    # memory_leak_check Seems to be failing memory leak check intermitently, see BS-534
):
    """
    checks that a regex starting and ending with % is correct
    """
    check_query(
        f"select case when A {like_expression} '%e%' then 1 else 0 end from table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
    )
    check_query(
        f"select case when A {like_expression} '%e.%' then 1 else 0 end from table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
    )


def check_wildcardless_like(pandas_code):
    """
    Checks that given pandas_code doesn't contain any contains
    code because the regular expression didn't contain any
    SQL wildcards.
    """
    assert ".str.contains" not in pandas_code


def check_start_percent_like(pandas_code):
    """
    Checks that given pandas_code doesn't uses endswith
    because the regular expression only included % at the beginning.
    """
    assert ".str.endswith" in pandas_code


def check_end_percent_like(pandas_code):
    """
    Checks that given pandas_code doesn't uses startswith
    because the regular expression only included % at the beginning.
    """
    assert ".str.startswith" in pandas_code


def check_start_and_end_percent_like(pandas_code):
    """
    Checks that given pandas_code doesn't uses regex=False
    because the regular expression only included % at the beginning
    and end.
    """
    assert "regex=False" in pandas_code


@pytest.mark.slow
def test_utf_scalar(spark_info):
    check_query(
        "select 'ǖǘǚǜ'",
        {},
        spark_info,
        check_names=False,
    )
