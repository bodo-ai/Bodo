from importlib.metadata import PackageNotFoundError, version

import bodo_iceberg_connector.java_helpers as java_helpers
from bodo_iceberg_connector.errors import IcebergError, IcebergJavaError
from bodo_iceberg_connector.parquet_info import (
    bodo_connector_get_parquet_file_list,
    bodo_connector_get_parquet_info,
)
from bodo_iceberg_connector.py4j_support import launch_jvm
from bodo_iceberg_connector.schema import (
    get_iceberg_runtime_schema,
    get_iceberg_typing_schema,
    get_typing_info,
)
from bodo_iceberg_connector.schema_helper import pyarrow_to_iceberg_schema_str
from bodo_iceberg_connector.table_info import (
    bodo_connector_get_current_snapshot_id,
)
from bodo_iceberg_connector.write import commit_merge_cow, commit_write

# ----------------------- Version Import from Metadata -----------------------
try:
    __version__ = version("bodo-iceberg-connector")
except PackageNotFoundError:
    # Package is not installed
    pass
