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
    commit_write,
)

# ----------------------- Version Import from Metadata -----------------------
try:
    __version__ = version("bodo-iceberg-connector")
except PackageNotFoundError:
    # Package is not installed
    pass
