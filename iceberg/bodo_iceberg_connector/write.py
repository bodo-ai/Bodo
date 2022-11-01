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


def process_file_infos(
    fnames: List[str],
    all_metrics: Dict[str, Any],
    catalog_type,
    table_loc,
    db_name,
    table_name,
):
    """
    Cleanup and process file name and metric info to Java types
    before calling the Java commit functions

    Args:
        fnames: List of file paths (possibly relative or absolute)
        all_metrics: Metrics about written data to include in commit
        catalog_type: Type of catalog the Iceberg table is in
        table_loc: Warehouse location of data/ path (and files)
        db_name: Namespace / Database schema containing Iceberg table
        table_name: Name of Iceberg table

    Returns:
        fnames_java: Java list of strings representing file names
        size_metric: Java list of longs representing file sizes in bytes
        count_metric: Java list of longs representing row/record count per file
    """

    fnames = [
        gen_file_loc(catalog_type, table_loc, db_name, table_name, name)
        for name in fnames
    ]
    fnames_java = convert_list_to_java(fnames)
    size_metric = convert_list_to_java(
        [TypeInt(x, JavaType.PRIMITIVE_LONG) for x in all_metrics["size"]]
    )
    count_metric = convert_list_to_java(
        [TypeInt(x, JavaType.PRIMITIVE_LONG) for x in all_metrics["record_count"]]
    )

    return fnames_java, size_metric, count_metric


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

    Args:
        conn_str: Connection string to catalog
        db_name: Namespace containing the table written to
        table_name: Name of table written to
        fnames: Names of Parquet file that need to be commited in Iceberg
        all_metrics: Metrics about written data to include in commit
        iceberg_schema_id: Known Schema ID when files were written
        pa_schema: Arrow Schema of written data
        partition_spec: Iceberg-based partitioning schema of data
        sort_order: Iceberg-based sorting of data
        mode: Method of Iceberg write (`create`, `replace`, `append`)

    Returns:
        bool: Whether the action was successfully commited or not
    """
    catalog_type, _ = parse_conn_str(conn_str)
    handler = get_java_table_handler(conn_str, catalog_type, db_name, table_name)
    fnames_java, size_metric, count_metric = process_file_infos(
        fnames, all_metrics, catalog_type, table_loc, db_name, table_name
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
            print("Error during Iceberg table creation: ", e)
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
            print("Error during Iceberg table replace: ", e)
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
            print("Error during Iceberg table append: ", e)
            return False

    return True


def commit_merge_cow(
    conn_str: str,
    db_name: str,
    table_name: str,
    table_loc: str,
    old_fnames: List[str],
    new_fnames: List[str],
    all_metrics: Dict[str, List[Any]],
    snapshot_id: int,
):
    """
    Commit the write step of MERGE INTO using copy-on-write rules

    Args:
        conn_str: Connection string to Iceberg catalog
        db_name: Namespace / Database schema of table
        table_name: Name of Iceberg table to write to
        table_loc: Warehouse location of data/ folder for Iceberg table
        old_fnames: List of old file paths to invalidate in commit
        new_fnames: List of written files to replace old_fnames
        all_metrics: Iceberg metrics for new_fnames
        snapshot_id: Expected current snapshot ID

    Returns:
        True if commit suceeded, False otherwise
    """

    catalog_type, _ = parse_conn_str(conn_str)
    handler = get_java_table_handler(conn_str, catalog_type, db_name, table_name)

    old_fnames_java = convert_list_to_java(old_fnames)
    new_fnames_java, size_metric, count_metric = process_file_infos(
        new_fnames, all_metrics, catalog_type, table_loc, db_name, table_name
    )

    try:
        handler.mergeCOWTable(
            old_fnames_java, new_fnames_java, size_metric, count_metric, snapshot_id
        )
    except Py4JJavaError as e:
        print("Error during Iceberg MERGE INTO COW:", e)
        return False

    return True
