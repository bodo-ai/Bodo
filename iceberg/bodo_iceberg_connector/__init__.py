from importlib.metadata import PackageNotFoundError, version

import bodo_iceberg_connector.java_helpers as java_helpers
from bodo_iceberg_connector.errors import IcebergError, IcebergJavaError
from bodo_iceberg_connector.filter_to_java import (
    ColumnRef,
    FilterExpr,
    Scalar,
)
from bodo_iceberg_connector.py4j_support import launch_jvm, set_core_site_path
from bodo_iceberg_connector.write import (
    commit_merge_cow,
    commit_statistics_file,
    commit_write,
    delete_table,
    get_table_metadata_path,
)
from bodo_iceberg_connector.puffin import (
    BlobMetadata,
    StatisticsFile,
    get_old_statistics_file_path,
    table_columns_have_theta_sketches,
)

# ----------------------- Version Import from Metadata -----------------------
try:
    __version__ = version("bodo-iceberg-connector")
except PackageNotFoundError:
    # Package is not installed
    pass
