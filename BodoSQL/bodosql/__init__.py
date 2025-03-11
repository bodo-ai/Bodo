# Import BodoSQL types
from bodosql.bodosql_types.database_catalog import DatabaseCatalog, DatabaseCatalogType
from bodosql.bodosql_types.table_path import TablePath, TablePathType
from bodosql.bodosql_types.filesystem_catalog import (
    FileSystemCatalog,
    FileSystemCatalogType,
)
from bodosql.bodosql_types.snowflake_catalog import (
    SnowflakeCatalog,
    SnowflakeCatalogType,
)
from bodosql.bodosql_types.rest_catalog import (
    RESTCatalog,
    RESTCatalogType,
    get_REST_connection,
)
from bodosql.bodosql_types.glue_catalog import (
    GlueCatalog,
    GlueCatalogType,
    get_glue_connection,
)
from bodosql.bodosql_types.s3_tables_catalog import (
    S3TablesCatalog,
    S3TablesCatalogType,
    get_s3_tables_connection,
)

import bodosql.context_ext
import bodosql.ddl_ext
import bodosql.remove_pure_calls

# Import BodoSQL libs
import bodosql.libs.regex
import bodosql.libs.null_handling
import bodosql.libs.nullchecked_logical_operators
import bodosql.libs.sql_operators
import bodosql.libs.ntile_helper
import bodosql.libs.iceberg_merge_into

# Import BodoSQL kernels
import bodosql.kernels
import bodosql.kernels.lead_lag
import bodosql.kernels.lateral
import bodosql.kernels.listagg
import bodosql.kernels.crypto_funcs

from bodosql.context import BodoSQLContext

# ------------------------------ Version Import ------------------------------
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bodosql")
except PackageNotFoundError:
    # Package is not installed
    pass
