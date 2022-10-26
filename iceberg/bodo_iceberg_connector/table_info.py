# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Python APIs used to gather metadata information about the underlying Iceberg table.
"""

from bodo_iceberg_connector.catalog_conn import parse_conn_str
from bodo_iceberg_connector.errors import IcebergJavaError
from bodo_iceberg_connector.py4j_support import get_java_table_handler
from py4j.protocol import Py4JJavaError


def bodo_connector_get_current_snapshot_id(
    conn_str: str, db_name: str, table: str
) -> int:
    try:
        catalog_type, _ = parse_conn_str(conn_str)

        bodo_iceberg_table_reader = get_java_table_handler(
            conn_str,
            catalog_type,
            db_name,
            table,
        )
        snapshot_id = bodo_iceberg_table_reader.getSnapshotId()
    except Py4JJavaError as e:
        raise IcebergJavaError.from_java_error(e)

    return int(snapshot_id)
