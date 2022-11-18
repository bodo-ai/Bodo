import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import pyarrow as pa
from bodo_iceberg_connector.catalog_conn import gen_file_loc, parse_conn_str
from bodo_iceberg_connector.py4j_support import (
    convert_list_to_java,
    get_java_table_handler,
)
from bodo_iceberg_connector.schema_helper import arrow_to_iceberg_schema
from py4j.protocol import Py4JJavaError


@dataclass
class DataFileInfo:
    """
    Python Representation of the DataFileInfo class on Java's side
    Used for communicating between Python and Java by transforming
    objects into JSON form
    """

    path: str
    size: int
    record_count: int


def process_file_infos(
    fnames: List[str],
    all_metrics: Dict[str, Any],
    catalog_type,
    table_loc,
    db_name,
    table_name,
):
    """
    Process file name and metrics to a JSON string that can be transmitted
    and deserialized in the Java side.

    Args:
        fnames: List of file paths (possibly relative or absolute)
        all_metrics: Metrics about written data to include in commit
        catalog_type: Type of catalog the Iceberg table is in
        table_loc: Warehouse location of data/ path (and files)
        db_name: Namespace / Database schema containing Iceberg table
        table_name: Name of Iceberg table

    Returns:
        JSON String Representing DataFileInfo objects
    """

    fnames = [
        gen_file_loc(catalog_type, table_loc, db_name, table_name, name)
        for name in fnames
    ]

    file_infos = [
        asdict(DataFileInfo(fname, size, count))
        for fname, size, count in zip(
            fnames, all_metrics["size"], all_metrics["record_count"]
        )
    ]

    return json.dumps(file_infos)


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
    file_info_str = process_file_infos(
        fnames, all_metrics, catalog_type, table_loc, db_name, table_name
    )

    if mode == "create":
        assert (
            iceberg_schema_id is None
        ), "bodo_iceberg_connector Internal Error: Should never create existing table"
        try:
            handler.createOrReplaceTable(
                file_info_str,
                arrow_to_iceberg_schema(pa_schema),
                False,
            )
        except Py4JJavaError as e:
            print("Error during Iceberg table creation: ", e)
            return False

    elif mode == "replace":
        try:
            handler.createOrReplaceTable(
                file_info_str,
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
            handler.appendTable(file_info_str, iceberg_schema_id)
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
    new_file_info_str = process_file_infos(
        new_fnames, all_metrics, catalog_type, table_loc, db_name, table_name
    )

    try:
        handler.mergeCOWTable(old_fnames_java, new_file_info_str, snapshot_id)
    except Py4JJavaError as e:
        print("Error during Iceberg MERGE INTO COW:", e)
        return False

    return True
