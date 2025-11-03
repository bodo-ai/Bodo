import os

from bodosql.context import BodoSQLContext

from bodosql.bodosql_types.table_path import TablePath
from bodosql.bodosql_types.database_catalog import DatabaseCatalog
from bodosql.bodosql_types.filesystem_catalog import FileSystemCatalog
from bodosql.bodosql_types.snowflake_catalog import SnowflakeCatalog
    

use_cpp_backend = os.environ.get("BODOSQL_CPP_BACKEND", "0") != "0"
verbose_cpp_backend = os.environ.get("BODOSQL_VERBOSE_CPP_BACKEND", "0") != "0"
# Used for testing purposes to disable fallback to JIT backend
cpp_backend_no_fallback = os.environ.get("BODOSQL_CPP_BACKEND_NO_FALLBACK", "0") != "0"

# ------------------------------ Version Import ------------------------------
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bodosql")
except PackageNotFoundError:
    # Package is not installed
    pass
