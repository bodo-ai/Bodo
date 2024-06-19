# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Python APIs used to gather metadata information about the underlying Iceberg table.
"""

from py4j.protocol import Py4JError

from bodo_iceberg_connector.catalog_conn import parse_conn_str
from bodo_iceberg_connector.errors import IcebergJavaError
from bodo_iceberg_connector.py4j_support import get_java_table_handler


def bodo_connector_get_current_snapshot_id(
    conn_str: str, db_name: str, table: str
) -> int:
    catalog_type, _ = parse_conn_str(conn_str)

    try:
        bodo_iceberg_table_reader = get_java_table_handler(
            conn_str,
            catalog_type,
            db_name,
            table,
        )
        snapshot_id = bodo_iceberg_table_reader.getSnapshotId()
    except Py4JError as e:
        raise IcebergJavaError.from_java_error(e)

    return int(snapshot_id)


def bodo_connector_get_table_property(
    conn_str: str, db_name: str, table: str, property: str
) -> str:
    """Get the value of a property from table properties

    Args:
        conn_str (str): the connection string
        db_name (str): name of the database
        table (str): name of the table
        property (str): name of the property

    Raises:
        IcebergJavaError.from_java_error: Propogates errors from Py4J

    Returns:
        str: The value of the property
    """
    catalog_type, _ = parse_conn_str(conn_str)

    try:
        bodo_iceberg_table_reader = get_java_table_handler(
            conn_str,
            catalog_type,
            db_name,
            table,
        )

        value = bodo_iceberg_table_reader.getTableProperty(property)
    except Py4JError as e:
        raise IcebergJavaError.from_java_error(e)

    return value
