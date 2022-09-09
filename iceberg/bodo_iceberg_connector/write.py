from typing import Any, Dict, List, Optional

import pyarrow as pa
from bodo_iceberg_connector.catalog_conn import gen_file_loc, parse_conn_str
from bodo_iceberg_connector.py4j_support import (
    JavaType,
    TypeInt,
    convert_list_to_java,
    get_java_table_handler,
)
from bodo_iceberg_connector.schema_helper import arrow_to_iceberg_schema
from py4j.protocol import Py4JJavaError


def commit_write(
    conn_str: str,
    db_name: str,
    table_name: str,
    table_loc: str,
    fnames: List[str],
    all_metrics: Dict[str, List[Any]],
    iceberg_schema_id: Optional[int],
    pa_schema: pa.Schema,
    partition_spec: Optional[str],
    sort_order: Optional[str],
    mode: str,
):
    """
    Register a write action in an Iceberg catalog

    Parameters:
        conn_str: Connection string to catalog
        db_name: Namespace containing the table written to
        table_name: Name of table written to
        fnames: Names of Parquet file that need to be commited in Iceberg
        all_metrics: Metrics about written data to include in commit
        iceberg_schema_id: Known Schema ID when files were written
        pa_schema: Arrow Schema of written data
        partition_spec: Partitioning of data
        sort_order: Column(s) that the data was sorted by before writing
        mode: Method of Iceberg write (`create`, `replace`, `append`)

    Returns:
        bool: Whether the action was successfully commited or not
    """
    catalog_type, _ = parse_conn_str(conn_str)
    fnames = [
        gen_file_loc(catalog_type, table_loc, db_name, table_name, name)
        for name in fnames
    ]

    handler = get_java_table_handler(conn_str, catalog_type, db_name, table_name)
    fnames_java = convert_list_to_java(fnames)
    size_metric = convert_list_to_java(
        [TypeInt(x, JavaType.PRIMITIVE_LONG) for x in all_metrics["size"]]
    )
    count_metric = convert_list_to_java(
        [TypeInt(x, JavaType.PRIMITIVE_LONG) for x in all_metrics["record_count"]]
    )

    if mode == "create":
        assert (
            iceberg_schema_id is None
        ), "bodo_iceberg_connector Internal Error: Should never create existing table"
        try:
            handler.createOrReplaceTable(
                fnames_java,
                size_metric,
                count_metric,
                arrow_to_iceberg_schema(pa_schema),
                False,
            )
        except Py4JJavaError as e:
            print("Error during table creation:\n", e)
            return False

    elif mode == "replace":
        try:
            handler.createOrReplaceTable(
                fnames_java,
                size_metric,
                count_metric,
                arrow_to_iceberg_schema(pa_schema),
                True,
            )
        except Py4JJavaError as e:
            print("Error during table replace:\n", e)
            return False

    else:
        assert (
            mode == "append"
        ), "bodo_iceberg_connector Internal Error: Unknown write mode. Supported modes: 'create', 'replace', 'append'."
        assert iceberg_schema_id is not None

        try:
            handler.appendTable(
                fnames_java, size_metric, count_metric, iceberg_schema_id
            )
        except Py4JJavaError as e:
            print("Error during table append:\n", e)
            return False

    return True
