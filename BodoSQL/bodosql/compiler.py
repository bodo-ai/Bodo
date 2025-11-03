"""This module initializes the BodoSQL compiler extensions
and is necessary to import before using BodoSQL in JIT.
"""

# ruff: noqa

# Initialize JIT compiler
import bodo.decorators

import bodosql

# Import BodoSQL types
from bodosql.bodosql_types.database_catalog_ext import DatabaseCatalogType
from bodosql.bodosql_types.table_path_ext import TablePathType
from bodosql.bodosql_types.filesystem_catalog_ext import (
    FileSystemCatalogType,
)
from bodosql.bodosql_types.snowflake_catalog_ext import (
    SnowflakeCatalogType,
)
from bodosql.bodosql_types.rest_catalog_ext import (
    RESTCatalogType,
)
from bodosql.bodosql_types.glue_catalog_ext import (
    GlueCatalogType,
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


# Set top-level type aliases
bodosql.TablePathType = TablePathType
bodosql.DatabaseCatalogType = DatabaseCatalogType
bodosql.FileSystemCatalogType = FileSystemCatalogType
bodosql.SnowflakeCatalogType = SnowflakeCatalogType
bodosql.RESTCatalogType = RESTCatalogType
bodosql.GlueCatalogType = GlueCatalogType
