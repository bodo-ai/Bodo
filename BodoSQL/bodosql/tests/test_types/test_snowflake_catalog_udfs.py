# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Tests UDF operations with a Snowflake catalog.

This file uses several UDFs that are defined inside our partner Snowflake account.
In case any of these get deleted, here are commands that can be used to regenerate them,
including any dependent tables or views.

PLUS_ONE:

create or replace function PLUS_ONE(X NUMBER) returns NUMBER(38, 0) as
$$
    x + 1
$$

QUERY_FUNCTION:

create or replace function QUERY_FUNCTION() returns NUMBER as
$$
    select count(*) from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS
$$

TIMES_TWO:

create or replace function TIMES_TWO(N Number) returns NUMBER as
$$
    n * 2
$$

TIMES_TWO:

create or replace function TIMES_TWO(N Varchar) returns Varchar as
$$
   repeat(s, 2)
$$

ADD_DEFAULT_ONE:

create or replace function ADD_DEFAULT_ONE(X NUMBER, Y NUMBER DEFAULT 1) returns NUMBER as
$$
   x + y
$$

SECURE_ADD_ONE:

create or replace secure function SECURE_ADD_ONE(X NUMBER) returns NUMBER(38, 0) as
$$
    x + 1
$$

DOLLAR_STRING:

create or replace function DOLLAR_STRING() returns VARCHAR as '$$NICK\'s TEST FUNCTION$$'

BAD_ALIAS_FUNCTION:

create or replace function BAD_ALIAS_FUNCTION() returns NUMBER as
$$
   select 1 as OUTER
$$

VALIDATION_ERROR_FUNCTION:

create or replace function BAD_ALIAS_FUNCTION(S VARCHAR) returns VARBINARY as
$$
   COMPRESS(s, 'ZLIB')
$$

VALIDATION_ERROR_QUERY_FUNCTION:

create or replace function VALIDATION_ERROR_QUERY_FUNCTION(S VARCHAR) returns VARBINARY as
$$
    select COMPRESS(s || TO_VARCHAR(count(*)), 'ZLIB') from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION
$$

PLUS_ONE_WRAPPER:

create or replace function PLUS_ONE_WRAPPER(N NUMBER) returns NUMBER as
$$
    PLUS_ONE(n)
$$

QUERY_PARAM_FUNCTION:

create or replace function QUERY_PARAM_FUNCTION(N NUMBER) returns NUMBER as
$$
    SELECT COUNT(*) + n from snowflake_sample_data.tpch_SF1.region
$$


FILTER_QUERY_FUNC:

create or replace function FILTER_QUERY_FUNC(N NUMBER) returns NUMBER as
$$
    select count(*) from COUNT_TABLE where A < N
$$

ORDER_BY_QUERY_FUNC:

create or replace function ORDER_BY_QUERY_FUNC(N NUMBER) returns NUMBER as
$$
    select sum(A) from (select A from COUNT_TABLE ORDER BY DIV0(A, N) LIMIT 5)
$$

JOIN_COND_QUERY_FUNC:

create or replace function JOIN_COND_QUERY_FUNC(N NUMBER) returns NUMBER as
$$
    select count(*) from COUNT_TABLE t1 join COUNT_TABLE t2 on t1.A = DIV0(t2.A, N)
$$

PARTITION_BY_QUERY_FUNC:

create or replace function PARTITION_BY_QUERY_FUNC(N NUMBER) returns NUMBER as
$$
    select MAX(rn) from (select row_number() over (partition by DIV0(A, N) order by A) as rn from COUNT_TABLE)
$$

WINDOW_ORDER_BY_QUERY_FUNC:

create or replace function WINDOW_ORDER_BY_QUERY_FUNC(N NUMBER) returns NUMBER as
$$
    select MAX(rn) from (select row_number() over (partition by DIV0(A, 2) order by DIV0(A, N)) as rn from COUNT_TABLE)
$$

HAVING_QUERY_FUNC:

create or replace function HAVING_QUERY_FUNC(N NUMBER) returns NUMBER as
$$
    select SUM(m) from (select count(O_ORDERKEY) as m from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS group by O_ORDERKEY having COUNT(O_ORDERKEY) > n)
$$

QUALIFY_QUERY_FUNC:

create or replace function QUALIFY_QUERY_FUNC(N NUMBER) returns NUMBER as
$$
    select sum(A) from (select A from COUNT_TABLE qualify row_number() over (partition by A order by A) = N)
$$

create or replace function NESTED_SELECT_QUERY_FUNC(N NUMBER) returns NUMBER as
$$
    select sum(col) from (select A + N as col from COUNT_TABLE)
$$

UDF_VIEW:

create or replace view UDF_VIEW AS
    select A, QUERY_PARAM_FUNCTION(A) as B from COUNT_TABLE

NESTED_CORRELATION_FUNCTION:

create or replace function NESTED_CORRELATION_FUNCTION(N NUMBER) returns NUMBER as
$$
    select sum(QUERY_PARAM_FUNCTION(A) + N) from COUNT_TABLE
$$

REPEAT_FUNCTION:

create or replace function repeat_function(s VARCHAR) returns VARCHAR as
$$
repeat(s, 2)
$$

REPEAT_FUNCTION:

create or replace function repeat_function(s VARCHAR, n NUMBER) returns VARCHAR as
$$
repeat(s, n)
$$

In our azure account we have also defined the following UDFs:

PYTHON_ADD_ONE:

create or replace function PYTHON_ADD_ONE(N NUMBER)
returns NUMBER
language python
runtime_version = '3.10'
handler = 'addone_py'
as
$$
def addone_py(i):
  return i+1
$$

RET_INPUT_FUNCTION:

CREATE OR REPLACE FUNCTION RET_INPUT_FUNCTION(input_value DATE)
RETURNS DATE
LANGUAGE SQL
as
$$
SELECT input_value
$$;

"""
import datetime
import io

import numpy as np
import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import check_func, pytest_mark_one_rank, pytest_snowflake
from bodo.utils.typing import BodoError
from bodosql.tests.test_types.snowflake_catalog_common import (  # noqa
    azure_snowflake_catalog,
    test_db_snowflake_catalog,
)

pytestmark = pytest_snowflake


def test_expression_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that a basic expression UDF can be inlined.

    PLUS_ONE is manually defined inside TEST_DB.PUBLIC.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select PLUS_ONE(1) as OUTPUT"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    check_func(impl, (bc, query), py_output=pd.DataFrame({"OUTPUT": [2]}))


def test_query_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    can be inlined.

    QUERY_FUNCTION is manually defined inside TEST_DB.PUBLIC.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select QUERY_FUNCTION() as OUTPUT"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"OUTPUT": [1500000]}),
        check_dtype=False,
    )


def test_query_argument_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument can be inlined if the argument is a constant.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select QUERY_PARAM_FUNCTION(1) as OUTPUT"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"OUTPUT": [6]}),
        check_dtype=False,
    )


def test_query_column_argument_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a column argument can be inlined.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select QUERY_PARAM_FUNCTION(A) as OUTPUT from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"OUTPUT": np.arange(5, 15)}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_query_argument_filter_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument used in the filter can be inlined.

    FILTER_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in a filter
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select FILTER_QUERY_FUNC(A) as OUTPUT from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"OUTPUT": [0, 0, 5, 10, 15, 18, 21, 24, 25, 26]}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


@pytest_mark_one_rank
def test_unsupported_query_argument_order_by_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument and uses it in an order by is unsupported due
    to a gap in correlation.

    ORDER_BY_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in the order by clause.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select ORDER_BY_QUERY_FUNC(A) as OUTPUT from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Found correlation in plan",
    ):
        impl(bc, query)


@pytest.mark.skip("Correlation occurs because we can't prune an unused column.")
def test_query_argument_join_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes an argument used in the join condition can be inlined.

    JOIN_COND_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in the join condition
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select JOIN_COND_QUERY_FUNC(A) as OUTPUT from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"OUTPUT": [0, 5, 5, 5, 15, 15, 20, 45, 58, 105]}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


@pytest_mark_one_rank
def test_unsupported_query_argument_partition_by_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument and uses it in a partition by of a window function
    is unsupported due to a gap in correlation handling.

    PARTITION_BY_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in the partition by of a window function.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select PARTITION_BY_QUERY_FUNC(A) as OUTPUT from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Found correlation in plan",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_window_order_by_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument and uses it in a order by of a window function
    is unsupported due to a gap in correlation handling.

    WINDOW_ORDER_BY_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in the order by of a window function.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select WINDOW_ORDER_BY_QUERY_FUNC(A) as OUTPUT from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Found correlation in plan",
    ):
        impl(bc, query)


def test_query_argument_having_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes an argument used in having can be inlined.

    HAVING_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in a having clause.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select HAVING_QUERY_FUNC(A) as OUTPUT from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": [0, 1, 2, 3, 4] * 2})},
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame(
            {"OUTPUT": pd.array([1500000, None, None, None, None] * 2, dtype="Int64")}
        ),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


@pytest_mark_one_rank
def test_unsupported_query_argument_qualify_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument and uses it in qualify is unsupported due
    to a gap in correlation handling.

    QUALIFY_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in a qualify clause.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select QUALIFY_QUERY_FUNC(A) as OUTPUT from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Found correlation in plan",
    ):
        impl(bc, query)


def test_query_argument_nested_select_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument and uses it inside a nested select statement
    can be inlined.

    NESTED_SELECT_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses an inner nested select statement.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select NESTED_SELECT_QUERY_FUNC(A) as OUTPUT from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame(
            {"OUTPUT": [99, 126, 153, 180, 207, 234, 261, 288, 315, 342]}
        ),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


@pytest_mark_one_rank
def test_udf_multiple_definitions(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake with multiple definitions select the correct implementation.
    We only run this on 1 rank because we also test a type that cannot be cast.

    TIMES_TWO is manually defined inside TEST_DB.PUBLIC twice, once on strings and
    once on numbers.
    """

    def impl(bc, query):
        return bc.sql(query)

    local_table = pd.DataFrame(
        {
            # Number
            "A": np.arange(10),
            # String
            "B": [str(i) for i in range(10)],
            # Double which casts to string. Note all types prioritize
            # string over number except for number.
            "C": [np.float64(i) for i in np.arange(1, 11)],
            # Binary which cannot be supported
            "D": [str(i).encode("utf-8") for i in range(10)],
        }
    )
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": local_table},
        catalog=test_db_snowflake_catalog,
    )
    query1 = "select TIMES_TWO(A) as OUTPUT from local_table"
    py_output1 = pd.DataFrame({"OUTPUT": local_table["A"] * 2})
    check_func(
        impl,
        (bc, query1),
        py_output=py_output1,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )
    query2 = "select TIMES_TWO(B) as OUTPUT from local_table"
    py_output2 = pd.DataFrame({"OUTPUT": local_table["B"] + local_table["B"]})
    check_func(
        impl,
        (bc, query2),
        py_output=py_output2,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )
    query3 = "select TIMES_TWO(C) as OUTPUT from local_table"
    # Bodo always uses 6 decimal places right now.
    float_to_str = local_table["C"].astype(str) + "00000"
    py_output3 = pd.DataFrame({"OUTPUT": float_to_str + float_to_str})
    check_func(
        impl,
        (bc, query3),
        py_output=py_output3,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )
    query4 = "select TIMES_TWO(D) as OUTPUT from local_table"
    with pytest.raises(
        BodoError,
        match="No match found for function signature TIMES_TWO\\(<BINARY>\\)",
    ):
        bodo.jit(impl)(bc, query4)


@pytest_mark_one_rank
def test_unsupported_udf_defaults(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with defaults gives a message they aren't supported.
    This is because we can't find the default values yet.

    ADD_DEFAULT_ONE is manually defined inside TEST_DB.PUBLIC with a default value.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select ADD_DEFAULT_ONE(1)"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match='Function "TEST_DB"\\."PUBLIC"\\."ADD_DEFAULT_ONE" uses default arguments, which are not supported on Snowflake UDFs because the default values cannot be found in Snowflake metadata\\. Missing argument\\(s\\): Y',
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_secure_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake secure UDFs, which we can't support, throws an appropriate
    error message.

    SECURE_ADD_ONE is manually defined inside TEST_DB.PUBLIC as secure.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select SECURE_ADD_ONE(1)"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match='Function "TEST_DB"\\."PUBLIC"\\."SECURE_ADD_ONE" contains an unsupported feature. Error message: "Bodo does not support secure UDFs',
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_python_udf(azure_snowflake_catalog, memory_leak_check):
    """
    Test that a Python UDF, which we can't support, throws an appropriate
    error message. We must test with Azure because our partner account doesn't
    have permission to use Python.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select PYTHON_ADD_ONE(1)"
    bc = bodosql.BodoSQLContext(catalog=azure_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match='Function "TEST_DB"\\."PUBLIC"\\."PYTHON_ADD_ONE" contains an unsupported feature. Error message: "Unsupported source language. Bodo supports SQL UDFs and has limited support for JavaScript UDFs\\. Source language found was PYTHON\\."',
    ):
        impl(bc, query)


def test_expression_udf_with_provided_defaults(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs that allow defaults can be inlined if values are provided
    for the defaults.

    ADD_DEFAULT_ONE is manually defined inside TEST_DB.PUBLIC with a default value.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select ADD_DEFAULT_ONE(1, 2) as OUTPUT"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    check_func(impl, (bc, query), py_output=pd.DataFrame({"OUTPUT": [3]}))


def test_expression_udf_with_named_args(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with named args can be inlined.

    ADD_DEFAULT_ONE is manually defined inside TEST_DB.PUBLIC with a default value.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select ADD_DEFAULT_ONE(Y => 1, X => 2) as OUTPUT"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    check_func(impl, (bc, query), py_output=pd.DataFrame({"OUTPUT": [3]}))


def test_udf_dollar_string(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a dollar string can be inlined.

    DOLLAR_STRING is manually defined inside TEST_DB.PUBLIC with a $$ quoted string
    as the body.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select DOLLAR_STRING() as OUTPUT"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    check_func(
        impl, (bc, query), py_output=pd.DataFrame({"OUTPUT": ["NICK'S TEST FUNCTION"]})
    )


@pytest_mark_one_rank
def test_unsupported_udf_parsing(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with contents that can't be parsed give an error indicating this.

    BAD_ALIAS_FUNCTION is manually defined inside TEST_DB.PUBLIC to use the alias OUTER,
    which our parser does not support yet.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select BAD_ALIAS_FUNCTION()"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.BAD_ALIAS_FUNCTION\\.\nCaused by: Failed to parse the function either as an Expression or as a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_udf_validation(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with contents that can't be validated give an error indicating this.

    VALIDATION_ERROR_FUNCTION is manually defined inside TEST_DB.PUBLIC to use COMPRESS,
    which we don't support.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select VALIDATION_ERROR_FUNCTION('abc')"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.VALIDATION_ERROR_FUNCTION\\.\nCaused by: .* No match found for function signature COMPRESS\\(<CHARACTER>, <CHARACTER>\\)",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_udf_query_validation(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with contents that can't be validated give an error indicating this
    even for functions with a query function body (e.g. SELECT).

    VALIDATION_ERROR_QUERY_FUNCTION is manually defined inside TEST_DB.PUBLIC to use COMPRESS,
    which we don't support and has a query function body (e.g. SELECT).
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select VALIDATION_ERROR_QUERY_FUNCTION('abc')"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.VALIDATION_ERROR_QUERY_FUNCTION\\.\nCaused by: .* No match found for function signature COMPRESS\\(<CHARACTER>, <CHARACTER>\\)",
    ):
        impl(bc, query)


def test_nested_expression_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that a basic expression UDF that calls another expression UDF
    can be inlined.

    PLUS_ONE_WRAPPER is manually defined inside TEST_DB.PUBLIC
    and simply calls PLUS_ONE.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select PLUS_ONE_WRAPPER(1) as OUTPUT"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    check_func(impl, (bc, query), py_output=pd.DataFrame({"OUTPUT": [2]}))


def test_repeated_nested_expression_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that a basic expression UDF that calls another expression UDF
    can be inlined and called multiple times.

    PLUS_ONE_WRAPPER is manually defined inside TEST_DB.PUBLIC
    and simply calls PLUS_ONE.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select PLUS_ONE_WRAPPER(1) as OUTPUT1, PLUS_ONE_WRAPPER(2) as OUTPUT2"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    check_func(
        impl, (bc, query), py_output=pd.DataFrame({"OUTPUT1": [2], "OUTPUT2": [3]})
    )


def test_query_column_expression_argument_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes an expression computed on a column can be inlined.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select QUERY_PARAM_FUNCTION(A + 1) as OUTPUT from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"OUTPUT": np.arange(6, 16)}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_query_multi_column_expression_argument_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes an expression consisting of multiple columns can be inlined.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select QUERY_PARAM_FUNCTION(A + B) AS OUTPUT from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10), "B": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"OUTPUT": np.arange(5, 24, 2)}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_nested_udf_calls(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes an expression consisting of nested calls to a function
    can be inlined.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select QUERY_PARAM_FUNCTION(QUERY_PARAM_FUNCTION(A + B) + C) as OUTPUT from local_table"
    bc = bodosql.BodoSQLContext(
        {
            "LOCAL_TABLE": pd.DataFrame(
                {"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)}
            )
        },
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"OUTPUT": np.arange(10, 40, 3)}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_filter_udf_calls(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that is used in a filter can be inlined.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select B from local_table where QUERY_PARAM_FUNCTION(A) > 10"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10), "B": np.arange(9, -1, -1)})},
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"B": np.arange(3, -1, -1)}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_order_by_udf_calls(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that in used in an order by can be inlined.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select * from local_table order by DIV0(A, QUERY_PARAM_FUNCTION(B)), B"
    table = pd.DataFrame({"A": np.arange(10), "B": np.arange(9, -1, -1)})
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": table},
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl,
        (bc, query),
        py_output=table,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "outer",
    [
        False,
        True,
    ],
)
@pytest_mark_one_rank
def test_unsupported_one_table_join_condition_udf_calls(
    test_db_snowflake_catalog, outer, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that in used in a join condition on 1 table fails with a message
    that a subquery cannot be expanded.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    def impl(bc, query):
        return bc.sql(query)

    join_type = "full outer" if outer else "inner"
    query = f"select * from LOCAL_TABLE1 t1 {join_type} join LOCAL_TABLE2 t2 on QUERY_PARAM_FUNCTION(t1.A) = t2.C"
    bc = bodosql.BodoSQLContext(
        {
            "LOCAL_TABLE1": pd.DataFrame(
                {"A": np.arange(10), "B": np.arange(9, -1, -1)}
            ),
            "LOCAL_TABLE2": pd.DataFrame({"C": np.arange(8)}),
        },
        catalog=test_db_snowflake_catalog,
    )
    if outer:
        with pytest.raises(
            BodoError,
            match="Found subquery in plan that could not be expanded",
        ):
            bodo.jit(impl)(bc, query)
    else:
        # Inner join is supported.
        table = pd.DataFrame(
            {"A": np.arange(3), "B": np.arange(9, 6, -1), "C": np.arange(5, 8)}
        )
        check_func(
            impl,
            (bc, query),
            py_output=table,
            check_dtype=False,
        )


@pytest.mark.parametrize(
    "outer",
    [
        False,
        True,
    ],
)
@pytest_mark_one_rank
def test_unsupported_each_table_join_condition_udf_calls(
    test_db_snowflake_catalog, outer, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that in used in a join condition on each table fails with a message
    that a subquery cannot be expanded.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    def impl(bc, query):
        return bc.sql(query)

    join_type = "full outer" if outer else "inner"
    query = f"select * from LOCAL_TABLE1 t1 {join_type} join LOCAL_TABLE2 t2 on QUERY_PARAM_FUNCTION(t1.A) = QUERY_PARAM_FUNCTION(t2.C)"
    bc = bodosql.BodoSQLContext(
        {
            "LOCAL_TABLE1": pd.DataFrame(
                {"A": np.arange(10), "B": np.arange(9, -1, -1)}
            ),
            "LOCAL_TABLE2": pd.DataFrame({"C": np.arange(8)}),
        },
        catalog=test_db_snowflake_catalog,
    )
    if outer:
        with pytest.raises(
            BodoError,
            match="Found subquery in plan that could not be expanded",
        ):
            bodo.jit(impl)(bc, query)
    else:
        # Inner join is supported.
        table = pd.DataFrame(
            {"A": np.arange(8), "B": np.arange(9, 1, -1), "C": np.arange(8)}
        )
        check_func(
            impl,
            (bc, query),
            py_output=table,
            check_dtype=False,
        )


@pytest.mark.parametrize(
    "outer",
    [
        False,
        True,
    ],
)
@pytest_mark_one_rank
def test_unsupported_both_tables_join_condition_udf_calls(
    test_db_snowflake_catalog, outer, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that in used in a join condition on with both tables fails with a message
    that a subquery cannot be expanded.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    join_type = "full outer" if outer else "inner"
    query = f"select * from LOCAL_TABLE1 t1 {join_type} join LOCAL_TABLE2 t2 on QUERY_PARAM_FUNCTION(t1.A + t2.C) = 10"
    bc = bodosql.BodoSQLContext(
        {
            "LOCAL_TABLE1": pd.DataFrame(
                {"A": np.arange(10), "B": np.arange(9, -1, -1)}
            ),
            "LOCAL_TABLE2": pd.DataFrame({"C": np.arange(8)}),
        },
        catalog=test_db_snowflake_catalog,
    )
    if outer:
        match_msg = "Found subquery in plan that could not be expanded"
    else:
        match_msg = "Found correlation in plan"
    with pytest.raises(
        BodoError,
        match=match_msg,
    ):
        impl(bc, query)


@pytest.mark.parametrize(
    "outer",
    [
        False,
        True,
    ],
)
def test_join_output_udf_calls(test_db_snowflake_catalog, outer, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that in used on both tables in the output of a join can be inlined.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    def impl(bc, query):
        return bc.sql(query)

    join_type = "full outer" if outer else "inner"
    query = f"select QUERY_PARAM_FUNCTION(t1.B + t2.C) as OUTPUT from LOCAL_TABLE1 t1 {join_type} join LOCAL_TABLE2 t2 on t1.A = t2.C"
    bc = bodosql.BodoSQLContext(
        {
            "LOCAL_TABLE1": pd.DataFrame(
                {"A": np.arange(10), "B": np.arange(9, -1, -1)}
            ),
            "LOCAL_TABLE2": pd.DataFrame({"C": np.arange(8)}),
        },
        catalog=test_db_snowflake_catalog,
    )
    # TODO [BSE-2593]: Support this query.
    # Note: This isn't working yet (gap in support using multiple namespaces
    # in producing correlation variables).
    if outer:
        py_output = pd.DataFrame(
            {
                "OUTPUT": pd.array(
                    [14, 14, 14, 14, 14, 14, 14, 14, None, None],
                    "Int64",
                ),
            }
        )
    else:
        py_output = pd.DataFrame({"OUTPUT": [14, 14, 14, 14, 14, 14, 14, 14]})

    check_func(
        impl,
        (bc, query),
        py_output=py_output,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_window_partition_by_udf_calls(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that in used in a window function's partition by can be inlined.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select row_number() OVER (partition by QUERY_PARAM_FUNCTION(C) order by B) as OUTPUT, A from LOCAL_TABLE"
    bc = bodosql.BodoSQLContext(
        {
            "LOCAL_TABLE": pd.DataFrame(
                {
                    "A": np.arange(10),
                    "B": np.arange(9, -1, -1),
                    "C": [1, 2, 3, 1, 2] * 2,
                }
            )
        },
        catalog=test_db_snowflake_catalog,
    )
    py_output = pd.DataFrame(
        {
            "OUTPUT": [4, 4, 2, 3, 3, 2, 2, 1, 1, 1],
            "A": np.arange(10),
        }
    )
    check_func(
        impl,
        (bc, query),
        py_output=py_output,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_window_order_by_udf_calls(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that in used in a window function's order by can be inlined.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select row_number() OVER (partition by C order by QUERY_PARAM_FUNCTION(B)) as OUTPUT, A from LOCAL_TABLE"
    bc = bodosql.BodoSQLContext(
        {
            "LOCAL_TABLE": pd.DataFrame(
                {
                    "A": np.arange(10),
                    "B": np.arange(9, -1, -1),
                    "C": [1, 2, 3, 1, 2] * 2,
                }
            )
        },
        catalog=test_db_snowflake_catalog,
    )
    py_output = pd.DataFrame(
        {
            "OUTPUT": [4, 4, 2, 3, 3, 2, 2, 1, 1, 1],
            "A": np.arange(10),
        }
    )
    check_func(
        impl,
        (bc, query),
        py_output=py_output,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_having_udf_calls(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that in used in a having clause can be inlined.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select count(A) as OUTPUT from LOCAL_TABLE group by A having COUNT(A) > QUERY_PARAM_FUNCTION(A)"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": [0, 1, 2, 3, 0, 1, 2, 1] * 3})},
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"OUTPUT": [6, 9]}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_qualify_udf_calls(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that in used in a qualify clause can be inlined.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select A from LOCAL_TABLE QUALIFY ROW_NUMBER() OVER (PARTITION BY B ORDER BY A) >= QUERY_PARAM_FUNCTION(A)"
    bc = bodosql.BodoSQLContext(
        {
            "LOCAL_TABLE": pd.DataFrame(
                {"A": [0, 1, 2, 3, 0, 1, 2, 1] * 3, "B": [1, 2, 1] * 8}
            )
        },
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"A": [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 3]}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_view_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that a view access that calls a SNOWFLAKE_UDF inlines
    the view and its UDF.

    UDF_VIEW is manually defined inside TEST_DB.PUBLIC and calls
    QUERY_PARAM_FUNCTION which is manually defined inside TEST_DB.PUBLIC.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select * from UDF_VIEW"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    py_output = pd.DataFrame(
        {
            "A": [1, 2, 3] * 5 + [4, 5, 6] * 3 + [7, 8, 9],
            "B": [6, 7, 8] * 5 + [9, 10, 11] * 3 + [12, 13, 14],
        }
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    # Ensure we always inline views
    with set_logging_stream(logger, 2):
        check_func(impl, (bc, query), py_output=py_output, check_dtype=False)
        # Verify that NICK_BASE_TABLE is found in the logger message so the
        # view was inlined.
        check_logger_msg(stream, "COUNT_TABLE")


def test_udf_function_call_view_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that a calls into a SNOWFLAKE_UDF that contains a view that also
    calls a SNOWFLAKE_UDF inlines all views and UDFs.

    UDF_VIEW is manually defined inside TEST_DB.PUBLIC and calls
    QUERY_PARAM_FUNCTION which is manually defined inside TEST_DB.PUBLIC.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select QUERY_PARAM_FUNCTION(A + B) as OUTPUT from UDF_VIEW"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    py_output = pd.DataFrame(
        {
            "OUTPUT": [12, 14, 16] * 5 + [18, 20, 22] * 3 + [24, 26, 28],
        }
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    # Ensure we always inline views
    with set_logging_stream(logger, 2):
        check_func(impl, (bc, query), py_output=py_output, check_dtype=False)
        # Verify that COUNT_TABLE is found in the logger message so the
        # view was inlined.
        check_logger_msg(stream, "COUNT_TABLE")


def test_nested_correlation_function_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that a calls into a SNOWFLAKE_UDF that contains a call to another
    SNOWFLAKE_UDF requiring a correlation inlines successfully.

    NESTED_CORRELATION_FUNCTION is manually defined inside TEST_DB.PUBLIC and calls
    QUERY_PARAM_FUNCTION from another table, which is manually defined
    inside TEST_DB.PUBLIC.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select NESTED_CORRELATION_FUNCTION(A + B) as OUTPUT from LOCAL_TABLE"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10), "B": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame(
            {"OUTPUT": [234, 288, 342, 396, 450, 504, 558, 612, 666, 720]}
        ),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_multiple_definitions_udfs(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that a UDF with multiple definitions that are distinct in the total
    number of arguments can be successfully inlined.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    query1 = "select repeat_function('abc') as output"
    check_func(
        impl,
        (bc, query1),
        py_output=pd.DataFrame({"OUTPUT": ["abcabc"]}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )
    query2 = "select repeat_function('abc', 4) as output"
    check_func(
        impl,
        (bc, query2),
        py_output=pd.DataFrame({"OUTPUT": ["abcabcabcabc"]}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_dateadd_inline_bug(test_db_snowflake_catalog, memory_leak_check):
    """Test for a specific issue where inlining a UDF whose arguments included a dateadd
    function would cause a bug (BSE-2622).

    RET_INPUT_FUNCTION is manually defined inside of TEST_DB.PUBLIC.
    It takes one argument and returns it unchanged.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": [1] * 10})},
        catalog=test_db_snowflake_catalog,
    )

    query = "select RET_INPUT_FUNCTION(dateadd(DAY, A, '2024-02-08'::DATE)) as OUTPUT from LOCAL_TABLE\n"
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"OUTPUT": [datetime.date(2024, 2, 9)] * 10}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )
