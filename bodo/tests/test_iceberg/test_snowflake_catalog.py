"""Unit tests for the Snowflake Iceberg catalog without a live connection."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bodo.io.iceberg.catalog.snowflake import SnowflakeCatalog


@pytest.mark.parametrize(
    "namespace, schema_name",
    [
        ("ONLY_THIS_SCHEMA", "ONLY_THIS_SCHEMA"),
        (("ONLY_THIS_SCHEMA",), "ONLY_THIS_SCHEMA"),
        ("MY_DB.ONLY_THIS_SCHEMA", "MY_DB.ONLY_THIS_SCHEMA"),
        (("MY_DB", "ONLY_THIS_SCHEMA"), "MY_DB.ONLY_THIS_SCHEMA"),
    ],
)
def test_list_tables_scoped_to_schema(namespace, schema_name):
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    requested = {
        "database_name": "MY_DB",
        "schema_name": "ONLY_THIS_SCHEMA",
        "table_name": "EXPECTED_TABLE",
    }
    other = dict(requested, schema_name="OTHER_SCHEMA", table_name="OTHER_TABLE")

    def fetchall():
        query, params = cursor.execute.call_args.args
        if query == "SHOW ICEBERG TABLES IN SCHEMA (%s)" and params == (schema_name,):
            return [requested]
        return [requested, other]

    cursor.fetchall.side_effect = fetchall
    catalog = SimpleNamespace(
        connection=connection,
        _SnowflakeIdentifier=SnowflakeCatalog._SnowflakeIdentifier,
    )

    assert SnowflakeCatalog.list_tables(catalog, namespace) == [
        ("MY_DB", "ONLY_THIS_SCHEMA", "EXPECTED_TABLE")
    ]
    cursor.execute.assert_called_once_with(
        "SHOW ICEBERG TABLES IN SCHEMA (%s)", (schema_name,)
    )
