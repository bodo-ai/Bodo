"""
An implementation of the Snowflake-managed Iceberg tables catalog.
Vendored from a PyIceberg PR:
https://github.com/apache/iceberg-python/pull/687
Note that the PR is currently closed and abandoned
"""

from __future__ import annotations

import json
import logging
import os
import typing as pt
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.parse import urlparse

import pyarrow as pa
from boto3.session import Session
from pyiceberg.catalog import MetastoreCatalog, PropertiesUpdateSummary
from pyiceberg.exceptions import NoSuchTableError, TableAlreadyExistsError
from pyiceberg.io import (
    S3_ACCESS_KEY_ID,
    S3_REGION,
    S3_SECRET_ACCESS_KEY,
    S3_SESSION_TOKEN,
)
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import (
    CommitTableRequest,
    CommitTableResponse,
    StaticTable,
    Table,
    sorting,
)
from pyiceberg.typedef import EMPTY_DICT, Identifier, Properties
from snowflake.connector import DictCursor, SnowflakeConnection

import bodo.user_logging

logger = logging.getLogger(__name__)


METADATA_QUERY = "SELECT SYSTEM$GET_ICEBERG_TABLE_INFORMATION(%s) AS METADATA"


def escape_name(part_name: str) -> str:
    """Escape the part name to be used in Snowflake connector queries."""
    return (
        part_name
        if part_name.startswith('"') and part_name.endswith('"')
        else f'"{part_name}"'
    )


class SnowflakeCatalog(MetastoreCatalog):
    @dataclass(frozen=True, eq=True)
    class _SnowflakeIdentifier:
        """
        Snowflake follows the following format for identifiers:
        [database_name].[schema_name].[table_name]

        If the database_name is not provided, the schema_name is the first part of the identifier.
        Similarly, if the schema_name is not provided, the table_name is the first part of the identifier.

        This class is used to parse the identifier into its constituent parts and
        provide utility methods to work with them.
        """

        database: str | None
        schema: str | None
        table: str | None

        def __iter__(self) -> Iterator[str]:
            """
            Iterate of the non-None parts of the identifier.

            Returns:
                Iterator[str]: Iterator of the non-None parts of the identifier.
            """
            yield from filter(None, [self.database, self.schema, self.table])

        @classmethod
        def table_from_id(
            cls, identifier: str | Identifier
        ) -> SnowflakeCatalog._SnowflakeIdentifier:
            parts = SnowflakeCatalog.identifier_to_tuple(identifier)
            if len(parts) == 1:
                return cls(None, None, escape_name(parts[0]))
            elif len(parts) == 2:
                return cls(None, escape_name(parts[0]), escape_name(parts[1]))
            elif len(parts) == 3:
                return cls(
                    escape_name(parts[0]), escape_name(parts[1]), escape_name(parts[2])
                )

            raise ValueError(f"Invalid identifier: {identifier}")

        @classmethod
        def schema_from_string(
            cls, identifier: str
        ) -> SnowflakeCatalog._SnowflakeIdentifier:
            parts = identifier.split(".")
            if len(parts) == 1:
                return cls(None, parts[0], None)
            elif len(parts) == 2:
                return cls(parts[0], parts[1], None)

            raise ValueError(f"Invalid identifier: {identifier}")

        @property
        def table_name(self) -> str:
            return ".".join(self)

        @property
        def schema_name(self) -> str:
            return ".".join(self)

    open_metadata_requests: dict[str, str]

    def __init__(self, name: str, **properties: str):
        """
        params:
            name: Name of the catalog.
            user: Snowflake user.
            account: Snowflake account.
            authenticator: Snowflake authenticator.
            password: Snowflake password.
            private_key: Snowflake private key.
            role: Snowflake role.

        There are multiple ways to authenticate with Snowflake. We are supporting the following
        as of now:

        1. externalbrowser
        2. password
        3. private_key
        """

        super().__init__(name, **properties)

        params = {
            "user": properties["user"],
            "account": properties["account"],
        }

        if "authenticator" in properties:
            params["authenticator"] = properties["authenticator"]

        if "password" in properties:
            params["password"] = properties["password"]

        if "private_key" in properties:
            params["private_key"] = properties["private_key"]

        if "role" in properties:
            params["role"] = properties["role"]

        if "warehouse" in properties:
            params["warehouse"] = properties["warehouse"]

        if "database" in properties:
            params["database"] = properties["database"]

        if "schema" in properties:
            params["schema"] = properties["schema"]

        if "session_parameters" in properties:
            session_params = json.loads(properties["session_parameters"])
            params["session_parameters"] = session_params
            params.update(session_params)

        self.connection = SnowflakeConnection(**params)
        self.open_metadata_requests = {}

    def load_table(self, identifier: str | Identifier) -> Table:
        sf_identifier = SnowflakeCatalog._SnowflakeIdentifier.table_from_id(identifier)
        with self.connection.cursor(DictCursor) as cursor:
            try:
                if sf_identifier.table_name in self.open_metadata_requests:
                    cursor.get_results_from_sfqid(
                        self.open_metadata_requests[sf_identifier.table_name]
                    )
                    load_str = "prefetch"
                else:
                    cursor.execute(METADATA_QUERY, (sf_identifier.table_name,))
                    load_str = "normal load"

                if bodo.user_logging.get_verbose_level() >= 1:
                    bodo.user_logging.log_message(
                        "Snowflake-Managed Iceberg - Load Metadata",
                        f"Loading table `{sf_identifier.table_name}` from {load_str}",
                    )

                # Extract the metadata path from the output
                metadata = json.loads(cursor.fetchone()["METADATA"])["metadataLocation"]
            except Exception as e:
                raise NoSuchTableError(
                    f"Table {sf_identifier.table_name} not found"
                ) from e

        _fs_scheme = urlparse(metadata)
        _fs_props = {}

        if _fs_scheme.scheme == "s3":
            _fs_props.update(self._generate_s3_access_credentials())
        elif _fs_scheme.scheme == "gcs":
            assert os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"), (
                "GOOGLE_APPLICATION_CREDENTIALS not set. This is required for GCS access."
            )
        elif _fs_scheme.scheme == "azure":
            pass
        else:
            warnings.warn(f"Unsupported filesystem scheme: {_fs_scheme.scheme}")

        tbl = StaticTable.from_metadata(metadata, properties=_fs_props)
        tbl._identifier = (
            tuple(identifier.split(".")) if isinstance(identifier, str) else identifier
        )
        tbl.catalog = self

        return tbl

    def register_table(
        self, identifier: str | Identifier, metadata_location: str
    ) -> Table:
        query = "CREATE ICEBERG TABLE (%s) METADATA_FILE_PATH = (%s)"
        sf_identifier = SnowflakeCatalog._SnowflakeIdentifier.table_from_id(
            identifier if isinstance(identifier, str) else ".".join(identifier)
        )

        with self.connection.cursor(DictCursor) as cursor:
            try:
                cursor.execute(query, (sf_identifier.table_name, metadata_location))
            except Exception as e:
                raise TableAlreadyExistsError(
                    f"Table {sf_identifier.table_name} already exists"
                ) from e

        return self.load_table(identifier)

    def drop_table(self, identifier: str | Identifier) -> None:
        sf_identifier = SnowflakeCatalog._SnowflakeIdentifier.table_from_id(
            identifier if isinstance(identifier, str) else ".".join(identifier)
        )

        query = "DROP TABLE IF EXISTS (%s)"

        with self.connection.cursor(DictCursor) as cursor:
            cursor.execute(query, (sf_identifier.table_name,))

    def rename_table(
        self, from_identifier: str | Identifier, to_identifier: str | Identifier
    ) -> Table:
        sf_from_identifier = SnowflakeCatalog._SnowflakeIdentifier.table_from_id(
            from_identifier
            if isinstance(from_identifier, str)
            else ".".join(from_identifier)
        )
        sf_to_identifier = SnowflakeCatalog._SnowflakeIdentifier.table_from_id(
            to_identifier if isinstance(to_identifier, str) else ".".join(to_identifier)
        )

        query = "ALTER TABLE (%s) RENAME TO (%s)"

        with self.connection.cursor(DictCursor) as cursor:
            cursor.execute(
                query, (sf_from_identifier.table_name, sf_to_identifier.table_name)
            )

        return self.load_table(to_identifier)

    def commit_table(self, table_request: CommitTableRequest) -> CommitTableResponse:
        raise NotImplementedError

    def create_namespace(
        self, namespace: str | Identifier, properties: Properties = EMPTY_DICT
    ) -> None:
        sf_identifier = SnowflakeCatalog._SnowflakeIdentifier.schema_from_string(
            namespace if isinstance(namespace, str) else ".".join(namespace)
        )

        db_query = "CREATE DATABASE IF NOT EXISTS (%s)"
        schema_query = "CREATE SCHEMA IF NOT EXISTS (%s)"

        with self.connection.cursor(DictCursor) as cursor:
            if sf_identifier.database:
                cursor.execute(db_query, (sf_identifier.database,))
            cursor.execute(schema_query, (sf_identifier.schema_name,))

    def drop_namespace(self, namespace: str | Identifier) -> None:
        sf_identifier = SnowflakeCatalog._SnowflakeIdentifier.schema_from_string(
            namespace if isinstance(namespace, str) else ".".join(namespace)
        )

        sf_query = "DROP SCHEMA IF EXISTS (%s)"
        db_query = "DROP DATABASE IF EXISTS (%s)"

        with self.connection.cursor(DictCursor) as cursor:
            if sf_identifier.database:
                cursor.execute(db_query, (sf_identifier.database,))
            cursor.execute(sf_query, (sf_identifier.schema_name,))

    def list_tables(self, namespace: str | Identifier) -> list[Identifier]:
        sf_identifier = SnowflakeCatalog._SnowflakeIdentifier.schema_from_string(
            namespace if isinstance(namespace, str) else ".".join(namespace)
        )

        schema_query = "SHOW ICEBERG TABLES IN SCHEMA (%s)"
        db_query = "SHOW ICEBERG TABLES IN DATABASE (%s)"

        with self.connection.cursor(DictCursor) as cursor:
            if sf_identifier.database:
                cursor.execute(db_query, (sf_identifier.database,))
            else:
                cursor.execute(schema_query, (sf_identifier.schema,))

            return [
                (row["database_name"], row["schema_name"], row["table_name"])
                for row in cursor.fetchall()
            ]

    def list_namespaces(self, namespace: str | Identifier = ()) -> list[Identifier]:
        raise NotImplementedError

    def load_namespace_properties(self, namespace: str | Identifier) -> Properties:
        raise NotImplementedError

    def update_namespace_properties(
        self,
        namespace: str | Identifier,
        removals: set[str] | None = None,
        updates: Properties = EMPTY_DICT,
    ) -> PropertiesUpdateSummary:
        raise NotImplementedError

    def create_table(
        self,
        identifier: str | Identifier,
        schema: Schema | pa.Schema,
        location: str | None = None,
        partition_spec: PartitionSpec = UNPARTITIONED_PARTITION_SPEC,
        sort_order: sorting.SortOrder = sorting.UNSORTED_SORT_ORDER,
        properties: Properties = EMPTY_DICT,
    ) -> Table:
        raise NotImplementedError

    @staticmethod
    def _generate_s3_access_credentials():
        session = Session()

        credentials = session.get_credentials()
        current_credentials = credentials.get_frozen_credentials()

        return {
            S3_ACCESS_KEY_ID: current_credentials.access_key,
            S3_SECRET_ACCESS_KEY: current_credentials.secret_key,
            S3_SESSION_TOKEN: current_credentials.token,
            S3_REGION: os.environ.get("AWS_REGION", "us-east-1"),
        }

    def list_views(self, namespace: str | Identifier) -> list[Identifier]:
        raise NotImplementedError(
            "Views are not supported for Snowflake-managed Iceberg tables"
        )

    def drop_view(self, identifier: str | Identifier) -> None:
        raise NotImplementedError(
            "Views are not supported for Snowflake-managed Iceberg tables"
        )

    def view_exists(self, identifier: str | Identifier) -> bool:
        raise NotImplementedError(
            "Views are not supported for Snowflake-managed Iceberg tables"
        )

    def prefetch_metadata_paths(self, table_ids: pt.Sequence[str | Identifier]) -> None:
        """
        For a given list of tables (by their identifiers), start the initial
        fetch for the head metadata file for each table by submitting an async
        SQL query through the Snowflake connector.

        During load_table, if the catalog sees an open metadata file request,
        it will block for the async query to complete and use the result.
        """

        for table_id in table_ids:
            sf_identifier = SnowflakeCatalog._SnowflakeIdentifier.table_from_id(
                table_id
            )

            with self.connection.cursor(DictCursor) as cursor:
                cursor.execute_async(METADATA_QUERY, (sf_identifier.table_name,))
                query_id = cursor.sfqid
                assert query_id is not None

            self.open_metadata_requests[sf_identifier.table_name] = query_id
