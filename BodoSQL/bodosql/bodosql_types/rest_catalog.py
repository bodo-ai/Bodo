"""Python and JIT class for describing a REST Iceberg catalog. A REST
catalog contains all information needed to connect and use REST Iceberg catalog for organizing and modifying tables.
"""

from __future__ import annotations

import os

from bodo.io.iceberg.catalog import conn_str_to_catalog
from bodo.spawn.utils import run_rank0
from bodosql import DatabaseCatalog
from bodosql.imported_java_classes import JavaEntryPoint


def _create_java_REST_catalog(
    rest_uri: str,
    warehouse: str,
    token: str | None,
    credential: str | None,
    scope: str | None = None,
    default_schema: str | None = None,
):
    """
    Create a Java RESTCatalog object.
    Args:
        warehouse (str): The warehouse to connect to.
        rest_uri (str): The URI of the REST server.
        token (str): The token to use for authentication.
        credential (str): The credential to use for authentication.
        scope (str): The scope to use for authentication.
        default_schema (str): The default schema to use.
    Returns:
        JavaObject: A Java RESTCatalog object.
    """
    return JavaEntryPoint.buildIcebergRESTCatalog(
        rest_uri,
        warehouse,
        token,
        credential,
        scope,
        default_schema,
    )


class RESTCatalog(DatabaseCatalog):
    """
    Python class for storing the information
        needed to connect to a REST Iceberg catalog.
    """

    def __init__(
        self,
        warehouse: str,
        rest_uri: str,
        token: str | None = None,
        credential: str | None = None,
        scope: str | None = None,
        default_schema: str | None = None,
    ):
        """
        Create a REST catalog from a connection string to a REST catalog.
        Either a token or a credential must be provided.
        Args:
            warehouse (str): The warehouse to connect to.
            rest_uri (str): The URI of the REST server.
            token (str): The token to use for authentication.
            credential (str): The credential to use for authentication.
            scope (str): The scope to use for authentication.
            default_schema (str): The default schema to use.
        """
        self.warehouse = warehouse
        self.rest_uri = rest_uri
        self.token = token
        self.credential = credential
        self.scope = scope
        self.default_schema = default_schema
        if self.token is None:
            self.token = self.get_token()

        # Set the token as an environment variable so that it can be accessed at runtime
        # Used by the RESTConnectionType
        os.environ["__BODOSQL_REST_TOKEN"] = self.token

    def get_java_object(self):
        return _create_java_REST_catalog(
            self.rest_uri,
            self.warehouse,
            self.token,
            self.credential,
            self.scope,
            self.default_schema,
        )

    @run_rank0
    def get_token(self):
        """
        Get the token for the REST catalog from a pyiceberg catalog.
        """
        con_str = get_conn_str(
            self.rest_uri, self.warehouse, scope=self.scope, credential=self.credential
        )
        py_catalog = conn_str_to_catalog(con_str)
        return py_catalog._session.auth.auth_manager._token

    def __eq__(self, other):
        if not isinstance(other, RESTCatalog):
            return False
        return self.warehouse == other.warehouse


def get_conn_str(rest_uri, warehouse, scope=None, token=None, credential=None):
    """Get the connection string for a REST Iceberg catalog."""
    conn_str = f"iceberg+{rest_uri}?warehouse={warehouse}"
    if scope is not None:
        conn_str += f"&scope={scope}"
    if token is not None:
        conn_str += f"&token={token}"
    if credential is not None:
        conn_str += f"&credential={credential}"
    conn_str += "&sigv4=false"
    return conn_str
