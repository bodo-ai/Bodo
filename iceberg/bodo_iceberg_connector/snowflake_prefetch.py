import json

from bodo_iceberg_connector.py4j_support import get_snowflake_prefetch


def prefetch_sf_tables(conn_str: str, table_paths: list[str]) -> None:
    """
    Prefetch the metadata path for a list of Snowflake-managed Iceberg tables
    Used for internal BodoSQL code generation

    Args:
        conn_str (str): Snowflake connection string
        table_paths (list[str]): List of fully qualified table paths to prefetch
    """

    prefetch_inst = get_snowflake_prefetch(conn_str)
    # Py4J has trouble passing list of strings, so jsonify between Python and Java
    prefetch_inst.prefetchMetadataPaths(json.dumps(table_paths))
