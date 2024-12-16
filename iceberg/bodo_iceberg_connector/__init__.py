from importlib.metadata import PackageNotFoundError, version

import bodo_iceberg_connector.java_helpers as java_helpers
from bodo_iceberg_connector.errors import IcebergError, IcebergJavaError
from bodo_iceberg_connector.filter_to_java import (
    ColumnRef,
    FilterExpr,
    Scalar,
)
from bodo_iceberg_connector.parquet_info import (
    IcebergParquetInfo,
    get_bodo_parquet_info,
    bodo_connector_get_total_num_pq_files_in_table,
)
from bodo_iceberg_connector.py4j_support import launch_jvm, set_core_site_path
from bodo_iceberg_connector.schema import (
    get_iceberg_runtime_schema,
    get_iceberg_typing_schema,
    get_typing_info,
)
from bodo_iceberg_connector.catalog_conn import parse_conn_str as parse_iceberg_conn_str
from bodo_iceberg_connector.schema_helper import pyarrow_to_iceberg_schema_str
from bodo_iceberg_connector.table_info import (
    bodo_connector_get_current_snapshot_id,
)
from bodo_iceberg_connector.write import (
    commit_merge_cow,
    commit_statistics_file,
    commit_write,
    delete_table,
    fetch_puffin_metadata,
    get_table_metadata_path,
    get_schema_with_init_field_ids,
    remove_transaction,
    start_write,
)
from bodo_iceberg_connector.puffin import (
    BlobMetadata,
    StatisticsFile,
    get_old_statistics_file_path,
    table_columns_have_theta_sketches,
    table_columns_enabled_theta_sketches,
)
from bodo_iceberg_connector.snowflake_prefetch import prefetch_sf_tables

# ----------------------- Version Import from Metadata -----------------------
try:
    __version__ = version("bodo-iceberg-connector")
except PackageNotFoundError:
    # Package is not installed
    pass
