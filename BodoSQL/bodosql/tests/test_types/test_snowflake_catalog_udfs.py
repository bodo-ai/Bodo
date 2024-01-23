# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Tests UDF operations with a Snowflake catalog.
"""
import numpy as np
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


@pytest_mark_one_rank
def test_unsupported_query_column_argument_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a column argument gives a message that they aren't supported yet,
    which should differ from the default "access" issues.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select QUERY_PARAM_FUNCTION(A) from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_filter_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    FILTER_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in a filter
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select FILTER_QUERY_FUNC(A) from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.FILTER_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_order_by_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    ORDER_BY_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in the order by clause.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select ORDER_BY_QUERY_FUNC(A) from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.ORDER_BY_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_join_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    JOIN_COND_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in the join condition
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select JOIN_COND_QUERY_FUNC(A) from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.JOIN_COND_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_partition_by_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    PARTITION_BY_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in the partition by of a window function.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select PARTITION_BY_QUERY_FUNC(A) from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.PARTITION_BY_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_window_order_by_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    WINDOW_ORDER_BY_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in the order by of a window function.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select WINDOW_ORDER_BY_QUERY_FUNC(A) from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.WINDOW_ORDER_BY_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_having_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    HAVING_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in a having clause.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select HAVING_QUERY_FUNC(A) from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.HAVING_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_qualify_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    QUALIFY_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in a qualify clause.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select QUALIFY_QUERY_FUNC(A) from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUALIFY_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_nested_select_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    NESTED_SELECT_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses an inner nested select statement.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select NESTED_SELECT_QUERY_FUNC(A) from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.NESTED_SELECT_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_filter_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    FILTER_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in a filter
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select FILTER_QUERY_FUNC(A) from table1"
    bc = bodosql.BodoSQLContext(
        {"table1": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.FILTER_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with arguments whose function bodies contain a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_order_by_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    ORDER_BY_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in the order by clause.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select ORDER_BY_QUERY_FUNC(A) from table1"
    bc = bodosql.BodoSQLContext(
        {"table1": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.ORDER_BY_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with arguments whose function bodies contain a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_join_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    JOIN_COND_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in the join condition
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select JOIN_COND_QUERY_FUNC(A) from table1"
    bc = bodosql.BodoSQLContext(
        {"table1": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.JOIN_COND_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with arguments whose function bodies contain a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_partition_by_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    PARTITION_BY_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in the partition by of a window function.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select PARTITION_BY_QUERY_FUNC(A) from table1"
    bc = bodosql.BodoSQLContext(
        {"table1": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.PARTITION_BY_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with arguments whose function bodies contain a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_window_order_by_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    WINDOW_ORDER_BY_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in the order by of a window function.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select WINDOW_ORDER_BY_QUERY_FUNC(A) from table1"
    bc = bodosql.BodoSQLContext(
        {"table1": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.WINDOW_ORDER_BY_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with arguments whose function bodies contain a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_having_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    HAVING_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in a having clause.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select HAVING_QUERY_FUNC(A) from table1"
    bc = bodosql.BodoSQLContext(
        {"table1": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.HAVING_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with arguments whose function bodies contain a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_qualify_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    QUALIFY_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses it in a qualify clause.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select QUALIFY_QUERY_FUNC(A) from table1"
    bc = bodosql.BodoSQLContext(
        {"table1": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUALIFY_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with arguments whose function bodies contain a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_argument_nested_select_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes a argument gives a message that
    they aren't supported yet, which should differ from the default
    "access" issues.

    NESTED_SELECT_QUERY_FUNC is manually defined inside TEST_DB.PUBLIC to take
    one argument and uses an inner nested select statement.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select NESTED_SELECT_QUERY_FUNC(A) from table1"
    bc = bodosql.BodoSQLContext(
        {"table1": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.NESTED_SELECT_QUERY_FUNC\\.\nCaused by: BodoSQL does not support Snowflake UDFs with arguments whose function bodies contain a query\\.",
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


@pytest_mark_one_rank
def test_unsupported_query_column_expression_argument_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes an expression computed on a column gives a message that
    they aren't supported yet, which should differ from the default "access" issues.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select QUERY_PARAM_FUNCTION(A + 1) from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_query_multi_column_expression_argument_udf(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes an expression consisting of multiple columns gives a message that
    they aren't supported yet, which should differ from the default "access" issues.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select QUERY_PARAM_FUNCTION(A + B) from local_table"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10), "B": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_nested_udf_calls(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that takes an expression consisting of nested calls to a function gives a message that
    they aren't supported yet, which should differ from the default "access" issues.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = (
        "select QUERY_PARAM_FUNCTION(QUERY_PARAM_FUNCTION(A + B) + C) from local_table"
    )
    bc = bodosql.BodoSQLContext(
        {
            "LOCAL_TABLE": pd.DataFrame(
                {"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)}
            )
        },
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_filter_udf_calls(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that is used in a filter gives a message that
    they aren't supported yet, which should differ from the default "access" issues.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select B from local_table where QUERY_PARAM_FUNCTION(A) > 10"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10), "B": np.arange(9, -1, -1)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_order_by_udf_calls(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that in used in an order by gives a message that
    they aren't supported yet, which should differ from the default "access" issues.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select * from local_table order by DIV0(A, QUERY_PARAM_FUNCTION(B)), B"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10), "B": np.arange(9, -1, -1)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


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
    that in used in a join condition on 1 table gives a message that
    they aren't supported yet, which should differ from the default "access" issues.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
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
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


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
    that in used in a join condition on each table gives a message that
    they aren't supported yet, which should differ from the default "access" issues.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
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
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


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
    that in used in a join condition on with both tables a message that
    they aren't supported yet, which should differ from the default "access" issues.

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
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest.mark.parametrize(
    "outer",
    [
        False,
        True,
    ],
)
@pytest_mark_one_rank
def test_unsupported_join_output_udf_calls(
    test_db_snowflake_catalog, outer, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that in used on both tables in the output of a join gives a message that
    they aren't supported yet, which should differ from the default "access" issues.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
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
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_window_partition_by_udf_calls(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that in used in a window function's partition by gives a message that
    they aren't supported yet, which should differ from the default "access" issues.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select row_number() OVER (partition by QUERY_PARAM_FUNCTION(C) order by B), A from LOCAL_TABLE"
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
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_window_order_by_udf_calls(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that in used in a window function's order by gives a message that
    they aren't supported yet, which should differ from the default "access" issues.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select row_number() OVER (partition by C order by QUERY_PARAM_FUNCTION(B)), A from LOCAL_TABLE"
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
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_having_udf_calls(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that in used in a having clause gives a message that
    they aren't supported yet, which should differ from the default "access" issues.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select count(A) from LOCAL_TABLE group by A having COUNT(A) > QUERY_PARAM_FUNCTION(A)"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": [0, 1, 2, 3, 0, 1, 2, 1] * 3})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_unsupported_qualify_udf_calls(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with a query function body (e.g. SELECT)
    that in used in a qualify clause gives a message that
    they aren't supported yet, which should differ from the default "access" issues.

    QUERY_PARAM_FUNCTION is manually defined inside TEST_DB.PUBLIC to take
    one argument.
    """

    @bodo.jit
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
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


def test_view_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that a view access that calls a SNOWFLAKE_UDF doesn't inline
    the view.

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
    check_func(impl, (bc, query), py_output=py_output, check_dtype=False)


@pytest_mark_one_rank
def test_udf_function_call_view_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that a calls into a SNOWFLAKE_UDF that contains a view that also
    calls a SNOWFLAKE_UDF gives a message that they aren't supported yet,
    which should differ from the default "access" issues.

    UDF_VIEW is manually defined inside TEST_DB.PUBLIC and calls
    QUERY_PARAM_FUNCTION which is manually defined inside TEST_DB.PUBLIC.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select QUERY_PARAM_FUNCTION(A + B) from UDF_VIEW"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.QUERY_PARAM_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)


@pytest_mark_one_rank
def test_nested_correlation_function_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that a calls into a SNOWFLAKE_UDF that contains a call to another
    SNOWFLAKE_UDF requiring a correlation gives a message that they aren't supported yet,
    which should differ from the default "access" issues.

    NESTED_CORRELATION_FUNCTION is manually defined inside TEST_DB.PUBLIC and calls
    QUERY_PARAM_FUNCTION from another table, which is manually defined
    inside TEST_DB.PUBLIC.
    """

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select NESTED_CORRELATION_FUNCTION(A + B) from LOCAL_TABLE"
    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": np.arange(10), "B": np.arange(10)})},
        catalog=test_db_snowflake_catalog,
    )
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.NESTED_CORRELATION_FUNCTION\\.\nCaused by: BodoSQL does not support Snowflake UDFs with column arguments whose function body contains a query\\.",
    ):
        impl(bc, query)
