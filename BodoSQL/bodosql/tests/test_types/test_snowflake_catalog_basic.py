"""
Tests basic components of the SnowflakeCatalog type both inside and outside a
direct BodoSQLContext. This file does not access Snowflake.
"""

from urllib.parse import urlencode

import pytest

import bodo
import bodosql
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError


@pytest.fixture(
    params=[
        bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
        ),
        bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
            {"role": "USERADMIN"},
        ),
    ]
)
def dummy_snowflake_catalogs(request):
    """
    List of table paths that should be supported.
    None of these actually point to valid data
    """
    return request.param


def test_snowflake_catalog_lower_constant(dummy_snowflake_catalogs, memory_leak_check):
    """
    Test lowering a constant snowflake catalog.
    """

    def impl():
        return dummy_snowflake_catalogs

    check_func(impl, ())


def test_snowflake_catalog_boxing(dummy_snowflake_catalogs, memory_leak_check):
    """
    Test boxing and unboxing a table path type.
    """

    def impl(snowflake_catalog):
        return snowflake_catalog

    check_func(impl, (dummy_snowflake_catalogs,))


def test_snowflake_catalog_constructor(memory_leak_check):
    """
    Test using the table path constructor from JIT.
    """

    def impl1():
        return bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
        )

    def impl2():
        return bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
            # Explicitly pass None
            None,
        )

    def impl3():
        return bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
            {"role": "USERADMIN"},
        )

    def impl4():
        return bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
            # Try passing an empty dictionary
            {},
        )

    check_func(impl1, ())
    check_func(impl2, ())
    check_func(impl3, ())
    # Note: Empty dictionary passed via args or literal map not supported yet.
    # [BE-3455]
    # check_func(impl4, ())


@pytest.mark.parametrize(
    "conn_str",
    [
        # Basic Connection Str
        "snowflake://myusername:mypassword@myaccount/mydatabase?warehouse=mywarehouse",
        # With Schema
        "snowflake://myusername:mypassword@myaccount/mydatabase/myschema?warehouse=mywarehouse",
        # Additional Connection Param. Note, order of params matter for test
        "snowflake://myusername:mypassword@myaccount/mydatabase/myschema?role=USERADMIN&warehouse=mywarehouse",
        # Missing Password
        "snowflake://myusername@myaccount/mydatabase?warehouse=mywarehouse",
    ],
)
def test_snowflake_catalog_from_conn_str(conn_str: str):
    c = bodosql.SnowflakeCatalog.from_conn_str(conn_str)

    params = c.connection_params.copy()
    params["warehouse"] = c.warehouse

    schema = params.pop("schema", None)
    password_str = f":{c.password}" if c.password != "" else ""
    schema_str = f"/{schema}" if schema is not None else ""

    params_sorted = sorted(params.items())
    expected = f"snowflake://{c.username}{password_str}@{c.account}/{c.database}{schema_str}?{urlencode(params_sorted)}"
    assert expected == conn_str, (
        "Connection String from SnowflakeCatalog does not match input arg"
    )


@pytest.mark.parametrize(
    "conn_str",
    [
        # Not a URL at all
        "test",
        # Non-Snowflake location
        "http://myaccount/mydatabase",
        # Invalid Connection Params
        "snowflake://myusername:mypassword@myaccount?test?test2",
    ],
)
def test_snowflake_catalog_from_conn_str_invalid_err(conn_str):
    """Test that invalid URIs fail when parsing"""
    with pytest.raises(ValueError, match="Invalid Snowflake Connection URI Provided"):
        bodosql.SnowflakeCatalog.from_conn_str(conn_str)


@pytest.mark.parametrize(
    "conn_str",
    [
        # Missing Username and Password
        "snowflake://myaccount/mydatabase",
        # Missing Username
        "snowflake://:mypassword@myaccount/mydatabase",
        # Missing Database
        "snowflake://myusername:mypassword@myaccount",
        # Missing Warehouse
        "snowflake://myusername:mypassword@myaccount/mydatabase",
    ],
)
def test_snowflake_catalog_from_conn_str_missing_err(conn_str):
    """Test that valid URIs fail due to missing contents"""
    with pytest.raises(ValueError, match="`conn_str` must contain a"):
        bodosql.SnowflakeCatalog.from_conn_str(conn_str)


def test_snowflake_catalog_from_conn_str_jit_err():
    def impl():
        return bodosql.SnowflakeCatalog.from_conn_str(
            "snowflake://user:pass@acc/db/schema"
        )

    with pytest.raises(
        BodoError,
        match="This constructor can not be called from inside of a Bodo-JIT function",
    ):
        bodo.jit(impl)()  # type: ignore
