import base64
import json
from dataclasses import asdict, dataclass
from typing import Any

from py4j.protocol import Py4JError

from bodo_iceberg_connector.catalog_conn import (
    gen_file_loc,
    normalize_loc,
    parse_conn_str,
)
from bodo_iceberg_connector.puffin import StatisticsFile
from bodo_iceberg_connector.py4j_support import (
    convert_list_to_java,
    get_catalog,
)


@dataclass
class DataFileInfo:
    """
    Python Representation of the DataFileInfo class on Java's side
    Used for communicating between Python and Java by transforming
    objects into JSON form
    """

    path: str
    file_size_in_bytes: int
    metrics: dict[str, Any]


class BytesEncoder(json.JSONEncoder):
    """
    JSON Encoder for bytes objects for lower/upper bound.
    """

    def default(self, o):
        if isinstance(o, bytes):
            return base64.b64encode(o).decode("ascii")
        else:
            return super().default(o)


def process_file_infos(
    fnames: list[str],
    file_size_bytes: list[int],
    metrics: list[dict[str, Any]],
    table_loc: str,
    db_name: str,
    table_name: str,
):
    """
    Process file name and metrics to a Java objects that can be passed
    to the connector.

    Args:
        fnames: List of file paths (possibly relative or absolute)
        metrics: Metrics about written data to include in commit
        table_loc: Warehouse location of data/ path (and files)
        db_name: Namespace / Database schema containing Iceberg table
        table_name: Name of Iceberg table

    Returns:
        JSON String Representing DataFileInfo objects
    """

    fnames = [gen_file_loc(table_loc, db_name, table_name, name) for name in fnames]
    file_infos = [
        asdict(DataFileInfo(fname, size, count))
        for fname, size, count in zip(fnames, file_size_bytes, metrics)
    ]
    return json.dumps(file_infos, cls=BytesEncoder)


def commit_write(
    txn_id: int,
    conn_str: str,
    db_name: str,
    table_name: str,
    table_loc: str,
    fnames: list[str],
    file_size_bytes: list[int],
    metrics: list[dict[str, Any]],
    iceberg_schema_id: int | None,
    mode: str,
):
    """
    Register a write action in an Iceberg catalog

    Args:
        txn_id: Transaction ID of the write action
        conn_str: Connection string to catalog
        db_name: Namespace containing the table written to
        table_name: Name of table written to
        table_loc: Warehouse location of data/ path (and files)
        fnames: Names of Parquet file that need to be committed in Iceberg
        file_size_bytes: Size of each file in bytes.
        metrics: Metrics about written data to include in commit
        iceberg_schema_id: Known Schema ID when files were written
        mode: Method of Iceberg write (`create`, `replace`, `append`)

    Returns:
        bool: Whether the action was successfully committed or not
    """
    catalog_type, _ = parse_conn_str(conn_str)
    handler = get_catalog(conn_str, catalog_type)
    file_info_str = process_file_infos(
        fnames, file_size_bytes, metrics, table_loc, db_name, table_name
    )

    if mode in ("create", "replace"):
        try:
            handler.commitCreateOrReplaceTable(txn_id, file_info_str)
        except Py4JError as e:
            print("Error during Iceberg table create/replace commit: ", e)
            return False

    else:
        assert mode == "append", (
            "bodo_iceberg_connector Internal Error: Unknown write mode. Supported modes: 'create', 'replace', 'append'."
        )
        assert iceberg_schema_id is not None

        try:
            handler.commitAppendTable(txn_id, file_info_str, iceberg_schema_id)
        except Py4JError as e:
            print("Error during Iceberg table append: ", e)
            return False
    return True


def commit_statistics_file(
    conn_str: str,
    table_id: str,
    snapshot_id: int,
    statistic_file_info: StatisticsFile,
):
    catalog_type, _ = parse_conn_str(conn_str)
    handler = get_catalog(conn_str, catalog_type)
    # Json encode the statistics file info
    statistic_file_info_str = json.dumps(asdict(statistic_file_info))
    handler.commitStatisticsFile(table_id, snapshot_id, statistic_file_info_str)


def commit_merge_cow(
    conn_str: str,
    db_name: str,
    table_name: str,
    table_loc: str,
    old_fnames: list[str],
    new_fnames: list[str],
    file_size_bytes: list[int],
    metrics: list[dict[str, Any]],
    snapshot_id: int,
):
    """
    Commit the write step of MERGE INTO using copy-on-write rules

    Args:
        conn_str: Connection string to Iceberg catalog
        db_name: Namespace / Database schema of table
        table_name: Name of Iceberg table to write to
        table_loc: Location of data/ folder for an Iceberg table
        old_fnames: List of old file paths to invalidate in commit
        new_fnames: List of written files to replace old_fnames
        file_size_bytes: Size of each file in bytes.
        metrics: Iceberg metrics for new_fnames
        snapshot_id: Expected current snapshot ID

    Returns:
        True if commit succeeded, False otherwise
    """

    catalog_type, _ = parse_conn_str(conn_str)
    old_fnames_java = convert_list_to_java(old_fnames)
    new_file_info_str = process_file_infos(
        new_fnames,
        file_size_bytes,
        metrics,
        table_loc,
        db_name,
        table_name,
    )

    try:
        handler = get_catalog(conn_str, catalog_type)
        handler.mergeCOWTable(
            db_name, table_name, old_fnames_java, new_file_info_str, snapshot_id
        )
    except Py4JError as e:
        # Note: Py4JError is the base class for all types of Py4j Exceptions.
        print("Error during Iceberg MERGE INTO COW:", e)
        return False

    return True


def delete_table(conn_str: str, db_name: str, table_name: str, purge: bool = True):
    """
    Delete an Iceberg table.
    args:
        conn_str: Iceberg connection string
        db_name: Database name
        table_name: Table name
        purge: If true, delete data/metadata files as well
    return: True if successful, False otherwise
    """
    catalog_type, _ = parse_conn_str(conn_str)
    handler = get_catalog(conn_str, catalog_type)
    try:
        return handler.deleteTable(db_name, table_name, purge)
    except Py4JError as e:
        print("Error during Iceberg table delete: ", e)
        return False


def get_table_metadata_path(conn_str: str, db_name: str, table_name: str):
    """
    Get the path to the current metadata file.
    """
    catalog_type, _ = parse_conn_str(conn_str)
    handler = get_catalog(conn_str, catalog_type)
    return normalize_loc(handler.getTableMetadataPath(db_name, table_name))
