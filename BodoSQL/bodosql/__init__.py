from .bodosql_types.database_catalog import DatabaseCatalog, DatabaseCatalogType
import bodosql.context_ext

# Import BodoSQL types
from bodosql.bodosql_types.table_path import TablePath, TablePathType
from bodosql.bodosql_types.database_catalog import DatabaseCatalog, DatabaseCatalogType
from bodosql.bodosql_types.filesystem_catalog import (
    FileSystemCatalog,
    FileSystemCatalogType,
)
from bodosql.bodosql_types.snowflake_catalog import (
    SnowflakeCatalog,
    SnowflakeCatalogType,
)

# Import BodoSQL libs
import bodosql.libs.regex
import bodosql.libs.null_handling
import bodosql.libs.nullchecked_logical_operators
import bodosql.libs.sql_operators
import bodosql.libs.ntile_helper
import bodosql.libs.iceberg_merge_into

# Import the library, throwing an error if it does not exist
import os

# TODO: put the generated library path in a global variable somewhere that is commonly accessible
GENERATED_LIB_FILE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "libs", "generated_lib.py"
)
if os.path.isfile(GENERATED_LIB_FILE_PATH):
    import bodosql.libs.generated_lib
else:
    raise Exception(
        "Error durring module import, did not find the generated library in the expected location: bodosql/libs/generated_lib.py"
    )

from bodosql.context import BodoSQLContext

# ------------------------------ Version Import ------------------------------
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bodosql")
except PackageNotFoundError:
    # Package is not installed
    pass
