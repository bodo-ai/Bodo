"""
Common helper functions and types for Iceberg support.
"""

from __future__ import annotations

import typing as pt
from urllib.parse import parse_qs, urlencode, urlparse

import numba
import pyarrow as pa
import requests
from numba.core import types
from numba.extending import overload

import bodo
from bodo.io.parquet_pio import getfs
from bodo.io.s3_fs import (
    create_iceberg_aws_credentials_provider,
    create_s3_fs_instance,
    get_region_from_creds_provider,
)
from bodo.utils.utils import BodoError, run_rank0

if pt.TYPE_CHECKING:  # pragma: no cover
    from pyarrow._fs import PyFileSystem


T = pt.TypeVar("T")
TVals = T | tuple["TVals", ...]


def flatten_tuple(x: tuple[TVals, ...]) -> tuple[T]:
    """
    Flatten a tuple of tuples into a single tuple. This is needed
    to handle nested tuples in the schema group identifier due to
    nested data types.
    """
    values = []
    for val in x:
        if isinstance(val, tuple):
            values.extend(flatten_tuple(val))
        else:
            values.append(val)
    return tuple(values)


def flatten_concatenation(list_of_lists: list[list[pt.Any]]) -> list[pt.Any]:
    """
    Helper function to flatten a list of lists into a
    single list.

    Ref: https://realpython.com/python-flatten-list/

    Args:
        list_of_lists (list[list[Any]]): List to flatten.

    Returns:
        list[Any]: Flattened list.
    """
    flat_list: list[pt.Any] = []
    for row in list_of_lists:
        flat_list += row
    return flat_list


FieldID: pt.TypeAlias = int | tuple["FieldID", ...]
FieldIDs: pt.TypeAlias = tuple[FieldID, ...]
FieldName: pt.TypeAlias = str | tuple["FieldName", ...]
FieldNames: pt.TypeAlias = tuple[FieldName, ...]
SchemaGroupIdentifier: pt.TypeAlias = tuple[FieldIDs, FieldNames]


# ===================================================================
# Must match the values in bodo_iceberg_connector/schema_helper.py
# ===================================================================
# This is the key used for storing the Iceberg Field ID in the
# metadata of the Arrow fields.
# Taken from: https://github.com/apache/arrow/blob/c23a097965b5c626cbc91b229c76a6c13d36b4e8/cpp/src/parquet/arrow/schema.cc#L245.
ICEBERG_FIELD_ID_MD_KEY = "PARQUET:field_id"

# PyArrow stores the metadata keys and values as bytes, so we need
# to use this encoded version when trying to access existing
# metadata in fields.
b_ICEBERG_FIELD_ID_MD_KEY = str.encode(ICEBERG_FIELD_ID_MD_KEY)
# ===================================================================


@run_rank0
def get_rest_catalog_config(conn: str) -> tuple[str, str, str] | None:
    """
    Get the configuration for a rest catalog connection string.
    @param conn: Iceberg connection string.
    @return: Tuple of uri, user_token, warehouse if successful, None otherwise (e.g. invalid connection string or not a rest catalog).
    """
    parsed_conn = urlparse(conn)
    if parsed_conn.scheme.lower() != "rest":
        return None
    parsed_conn = parsed_conn._replace(scheme="https")
    parsed_params = parse_qs(parsed_conn.query)
    # Clear the params
    parsed_conn = parsed_conn._replace(query="")
    uri = parsed_conn.geturl()

    user_token, credential, warehouse = (
        parsed_params.get("token"),
        parsed_params.get("credential"),
        parsed_params.get("warehouse"),
    )
    if user_token is not None:
        user_token = user_token[0]
    if warehouse is not None:
        warehouse = warehouse[0]
    # If we have a credential, we need to use it to get a user_token
    if credential is not None:
        credential = credential[0]
        client_id, client_secret = credential.split(":")
        user_token_request = requests.post(
            f"{uri}/v1/oauth/tokens",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if user_token_request.status_code != 200:
            raise BodoError(
                f"Unable to authenticate with {uri}. Please check your connection string."
            )
        user_token = user_token_request.json().get("access_token")

    if user_token is None:
        raise BodoError(
            f"Unable to authenticate with {uri}. Please check your connection string."
        )
    return uri, str(user_token), str(warehouse)


@numba.njit
def get_rest_catalog_fs(
    catalog_uri: str,
    bearer_token: str,
    warehouse: str,
    database_schema: str,
    table_name: str,
) -> pa.fs.FileSystem:
    """
    Get a filesystem object for the rest catalog.
    args:
        catalog_uri: URI of the rest catalog.
        bearer_token: Bearer token for authentication.
        warehouse: Warehouse name.
        database_schema: Schema the relevant table is in
        table_name: Name of the table
    """
    creds_provider = create_iceberg_aws_credentials_provider(
        catalog_uri, bearer_token, warehouse, database_schema, table_name
    )
    region = get_region_from_creds_provider(creds_provider)
    return create_s3_fs_instance(credentials_provider=creds_provider, region=region)


def get_iceberg_fs(
    protocol: str,
    conn: str,
    database_schema: str,
    table_name: str,
    pq_abs_path_file_list: list[str],
) -> PyFileSystem | pa.fs.FileSystem:
    rest_catalog_conf = get_rest_catalog_config(conn)
    if rest_catalog_conf is not None:
        uri, bearer_token, warehouse = rest_catalog_conf
        return get_rest_catalog_fs(
            uri, bearer_token, warehouse, database_schema, table_name
        )
    else:
        return getfs(
            pq_abs_path_file_list,
            protocol,
            storage_options=None,
            parallel=True,
        )


# ----------------------- Connection String Handling ----------------------- #


class IcebergConnectionType(types.Type):
    """
    Abstract base class for IcebergConnections
    """

    def __init__(self, name):  # pragma: no cover
        super().__init__(
            name=name,
        )

    def get_conn_str(self) -> str:
        raise NotImplementedError("IcebergConnectionType should not be instantiated")


def format_iceberg_conn(conn_str: str) -> str:
    """
    Determine if connection string points to an Iceberg database and reconstruct
    the correct connection string needed to connect to the Iceberg metastore
    """

    parse_res = urlparse(conn_str)
    if not conn_str.startswith("iceberg+glue") and parse_res.scheme not in (
        "iceberg",
        "iceberg+file",
        "iceberg+s3",
        "iceberg+thrift",
        "iceberg+http",
        "iceberg+https",
        "iceberg+snowflake",
        "iceberg+abfs",
        "iceberg+abfss",
        "iceberg+rest",
        "iceberg+arn",
    ):
        raise BodoError(
            "'con' must start with one of the following: 'iceberg://', 'iceberg+file://', "
            "'iceberg+s3://', 'iceberg+thrift://', 'iceberg+http://', 'iceberg+https://', 'iceberg+glue', 'iceberg+snowflake://', "
            "'iceberg+abfs://', 'iceberg+abfss://', 'iceberg+rest://', 'iceberg+arn'"
        )

    # Remove Iceberg Prefix when using Internally
    conn_str = conn_str.removeprefix("iceberg+").removeprefix("iceberg://")

    # Reformat Snowflake connection string to be iceberg-connector compatible
    if conn_str.startswith("snowflake://"):
        from bodo.io.snowflake import parse_conn_str

        conn_contents = parse_conn_str(conn_str)
        account: str = conn_contents.pop("account")
        # Flatten Session Parameters
        session_params = conn_contents.pop("session_parameters", {})
        conn_contents.update(session_params)
        # Remove Snowflake Specific Parameters
        conn_contents.pop("warehouse", None)
        conn_contents.pop("database", None)
        conn_contents.pop("schema", None)
        conn_str = (
            f"snowflake://{account}.snowflakecomputing.com/?{urlencode(conn_contents)}"
        )

    return conn_str


def format_iceberg_conn_njit(conn: str) -> str:  # type: ignore
    pass


@overload(format_iceberg_conn_njit)
def overload_format_iceberg_conn_njit(conn):  # pragma: no cover
    """
    Wrapper around format_iceberg_conn for strings
    Gets the connection string from conn_str attr for IcebergConnectionType

    Args:
        conn_str (str | IcebergConnectionType): connection passed in read_sql/read_sql_table/to_sql

    Returns:
        str: connection string without the iceberg(+*?) prefix
    """
    if isinstance(conn, (types.UnicodeType, types.StringLiteral)):

        def impl(conn):
            with bodo.no_warning_objmode(conn_str="unicode_type"):
                conn_str = format_iceberg_conn(conn)
            return conn_str

        return impl
    else:
        assert isinstance(
            conn, IcebergConnectionType
        ), f"format_iceberg_conn_njit: Invalid type for conn, got {conn}"

        def impl(conn):
            return conn.conn_str

        return impl
