# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Tests UDF operations with a Snowflake catalog.
"""
import pytest

import bodo
import bodosql
from bodo.tests.utils import pytest_snowflake
from bodo.utils.typing import BodoError
from bodosql.tests.test_types.snowflake_catalog_common import (  # noqa
    azure_snowflake_catalog,
    test_db_snowflake_catalog,
)

pytestmark = pytest_snowflake


def test_unsupported_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs give a message that they aren't supported yet,
    which should differ from the default "access" issues.

    PLUS_ONE is manually defined inside TEST_DB.PUBLIC.
    """
    if bodo.get_size() != 1:
        pytest.skip("This test is only designed for 1 rank")

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select PLUS_ONE(1)"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.PLUS_ONE\\. BodoSQL does not have support for Snowflake UDFs yet",
    ):
        impl(bc, query)


def test_unsupported_udf_multiple_definitions(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test that Snowflake UDFs give a message that they aren't supported yet because
    there are multiple definitions of the UDF.

    TIMES_TWO is manually defined inside TEST_DB.PUBLIC twice, once on strings and
    once on numbers.
    """
    if bodo.get_size() != 1:
        pytest.skip("This test is only designed for 1 rank")

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


def test_unsupported_udf_defaults(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake UDFs with defaults gives a message they aren't supported.
    This is because we can't find the default values yet.

    ADD_DEFAULT_ONE is manually defined inside TEST_DB.PUBLIC with a default value.
    """
    if bodo.get_size() != 1:
        pytest.skip("This test is only designed for 1 rank")

    @bodo.jit
    def impl(bc, query):
        return bc.sql(query)

    query = "select ADD_DEFAULT_ONE(1)"
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    with pytest.raises(
        BodoError,
        match="Unable to resolve function: TEST_DB\\.PUBLIC\\.ADD_DEFAULT_ONE\\. BodoSQL does not support Snowflake UDFs with default arguments\\.",
    ):
        impl(bc, query)


def test_unsupported_secure_udf(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that Snowflake secure UDFs, which we can't support, throws an appropriate
    error message.

    SECURE_ADD_ONE is manually defined inside TEST_DB.PUBLIC as secure.
    """
    if bodo.get_size() != 1:
        pytest.skip("This test is only designed for 1 rank")

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


def test_unsupported_python_udf(azure_snowflake_catalog, memory_leak_check):
    """
    Test that a Python UDF, which we can't support, throws an appropriate
    error message. We must test with Azure because our partner account doesn't
    have permission to use Python.
    """
    if bodo.get_size() != 1:
        pytest.skip("This test is only designed for 1 rank")

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
