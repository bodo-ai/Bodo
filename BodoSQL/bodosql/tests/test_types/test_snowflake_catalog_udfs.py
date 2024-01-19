# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Tests UDF operations with a Snowflake catalog.
"""
import pandas as pd
import pytest

import bodo
import bodosql
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
    gives a message that they aren't supported yet,
    which should differ from the default "access" issues.

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


@pytest_mark_one_rank
def test_unsupported_query_argument_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that they aren't supported yet,
    which should differ from the default "access" issues.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select QUERY_PARAM_FUNCTION(1)"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with arguments whose function bodies contain a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    gives a message that they aren't supported yet,
    which should differ from the default "access" issues.

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


@pytest_mark_one_rank
def test_unsupported_query_argument_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that they aren't supported yet,
    which should differ from the default "access" issues.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select QUERY_PARAM_FUNCTION(1)"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\. BodoSQL does not have support for Snowflake UDFs yet",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_udf_multiple_definitions(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs give a message that they aren't supported yet because
    there are multiple definitions of the UDF.

    TIMES_TWO is manually defined inside TEST_DB.PUBLIC twice, once on strings and
    once on numbers.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select TIMES_TWO(1)"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.TIMES_TWO\\. BodoSQL only supports Snowflake UDFs with a single definition\\. Found 2 definitions",
    ):
        impl(bc, query)


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
        match='Function "TEST_DB"\\."PUBLIC"\\."PYTHON_ADD_ONE" contains an unsupported feature. Error message: "Unsupported source language. Bodo only support SQL UDFs, but found PYTHON"',
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
    Test that Snowflake UDFs with a dollar string can supported.

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
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.VALIDATION_ERROR_QUERY_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with arguments whose function bodies contain a query\\.",
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
