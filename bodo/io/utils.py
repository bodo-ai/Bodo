"""
Utility functions for I/O operations.
This file should import JIT lazily to avoid slowing down non-JIT code paths.
"""

from __future__ import annotations

from pathlib import PureWindowsPath
from typing import Any
from urllib.parse import parse_qsl, urlparse


def parse_dbtype(con_str) -> tuple[str, str]:
    """
    Converts a constant string used for db_type to a standard representation
    for each database.
    """
    parseresult = urlparse(con_str)
    db_type = parseresult.scheme
    con_paswd = parseresult.password
    # urlparse skips oracle since its handle has _
    # which is not in `scheme_chars`
    # oracle+cx_oracle
    if con_str.startswith("oracle+cx_oracle://"):
        return "oracle", con_paswd
    if db_type == "mysql+pymysql":
        # Standardize mysql to always use "mysql"
        return "mysql", con_paswd

    # NOTE: if you're updating supported schemes here, don't forget
    # to update the associated error message in _run_call_read_sql_table

    if con_str.startswith("iceberg+glue") or parseresult.scheme in (
        "iceberg",
        "iceberg+file",
        "iceberg+s3",
        "iceberg+thrift",
        "iceberg+http",
        "iceberg+https",
    ):
        # Standardize iceberg to always use "iceberg"
        return "iceberg", con_paswd
    return db_type, con_paswd


def is_windows_path(path: str) -> bool:
    """
    Check if the given path is a Windows path (e.g. C:\\user\\data).
    """
    p = PureWindowsPath(path)

    # True if a typical Windows drive like "C:" or a UNC drive like "\\server\share"
    if p.drive:
        if len(p.drive) == 2 and p.drive[1] == ":":
            return True

        if p.drive.startswith("\\"):
            return True

    return False


def parse_snowflake_conn_str(
    conn_str: str, strict_parsing: bool = False
) -> dict[str, Any]:
    """
    Parse a Snowflake Connection URL into Individual Components,
    and save to a dict.

    Used for connecting to Snowflake from Pandas API, and passing
    a connection string to bodosql.SnowflakeCatalog

    Args:
        conn_str: Snowflake connection URL in the following format:
            snowflake://<user_login_name>:<password>@<account_identifier>/<database_name>/<schema_name>?warehouse=<warehouse_name>&role=<role_name>
            Required arguments include <user_login_name>, <password>, and
            <account_identifier>. Optional arguments include <database_name>,
            <schema_name>, <warehouse_name>, and <role_name>.
            Do not include the `snowflakecomputing.com` domain name as part of
            your account identifier. Snowflake automatically appends the domain
            name to your account identifier to create the required connection.
            (https://docs.snowflake.com/en/user-guide/sqlalchemy.html#connection-parameters)
        strict_parsing: Whether to throw an error or not if query parameters are invalid or
            incorrectly formatted. Only true for SnowflakeCatalog.from_conn_str

    Returns:
        Dictionary of contents. Some expected fields include:
            - user
            - password
            - account
            - port (optional, usually unspecified)
            - database
            - schema
            - session_parameters: dict[str, str] of special preset params
            - ...
    """

    u = urlparse(conn_str)
    if u.scheme != "snowflake":
        raise ValueError(
            f"Invalid Snowflake Connection URI Provided: Starts with {u.scheme}:// but expected to start with snowflake://.\n"
            "See https://docs.snowflake.com/developer-guide/python-connector/sqlalchemy#connection-parameters for more details on how to construct a valid connection URI"
        )

    params = {}
    if u.username:
        params["user"] = u.username
    if u.password:
        params["password"] = u.password
    if u.hostname:
        params["account"] = u.hostname
    if u.port:
        params["port"] = u.port
    if u.path:
        # path contains "database_name/schema_name"
        path = u.path
        if path.startswith("/"):
            path = path[1:]
        parts = path.split("/")
        if len(parts) == 2:
            database, schema = parts
        elif len(parts) == 1:  # pragma: no cover
            database = parts[0]
            schema = None
        else:  # pragma: no cover
            raise ValueError(
                f"Unexpected Snowflake connection string {conn_str}. Path is expected to contain database name and possibly schema"
            )
        params["database"] = database
        if schema:
            params["schema"] = schema
    if u.query:
        # query contains warehouse_name and role_name
        try:
            contents = parse_qsl(u.query, strict_parsing=strict_parsing)
        except ValueError as e:
            raise ValueError(f"Invalid Snowflake Connection URI Provided: {e.args[0]}")

        for key, val in contents:
            params[key] = val
            if key == "session_parameters":
                # Snowflake connector appends to session_parameters and
                # assumes it is a dictionary if provided. This is an existing
                # bug in SqlAlchemy/SnowflakeSqlAlchemy
                import json

                params[key] = json.loads(val)

    return params
