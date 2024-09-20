from enum import Enum

import requests
from cachetools import TTLCache, cached
from cachetools.keys import hashkey

import bodosql
from bodo_platform_utils.config import (
    CATALOG_PREFIX,
    SNOWFLAKE_PC_CATALOG_NAME,
    DEFAULT_SECRET_GROUP,
)
from bodo_platform_utils.secrets_utils import get

# constants
ICEBERG_AUTH_URL_SUFFIX = "/v1/oauth/tokens"


class CatalogType(Enum):
    SNOWFLAKE = "SNOWFLAKE"
    TABULAR = "TABULAR"
    GLUE = "GLUE"


class CatalogData:
    catalog: str
    warehouse: str = None
    database: str = None
    schema: str = None

    iceberg_volume: str = None
    iceberg_rest_url: str = None


def create_catalog(data: CatalogData):
    """Create the catalog from the catalog data.

    :param data: Data for catalog.
    """
    catalog = get_data(data.catalog)
    if catalog is None:
        raise ValueError("Catalog not found in the secret store.")

    # Get catalog type, default snowflake
    catalog_type_str = catalog.get("catalogType", str(CatalogType.SNOWFLAKE.value))
    catalog_type = CatalogType(catalog_type_str)

    if catalog_type == CatalogType.TABULAR:
        return create_tabular_catalog(catalog, data)

    if catalog_type == CatalogType.GLUE:
        return create_glue_catalog(catalog, data)

    # default to Snowflake for backward compatibility
    return create_snowflake_catalog(catalog, data)


def create_snowflake_catalog(catalog, data: CatalogData):
    """Create the Snowflake catalog from the catalog data.

    :param catalog: JSON catalog data loaded from secrets
    :param data: Optional data for catalog if provided will override data from secret store
    """
    warehouse = _get_warehouse(catalog, data.warehouse)
    database = _get_database(catalog, data.database)

    # Schema can be None for backwards compatibility
    schema = data.schema if data.schema else catalog.get("schema")

    # Iceberg volume can be None
    iceberg_volume = (
        data.iceberg_volume if data.iceberg_volume else catalog.get("icebergVolume")
    )

    # Create connection params
    connection_params = {"role": catalog["role"]} if "role" in catalog else {}
    if schema is not None:
        connection_params["schema"] = schema

    return bodosql.SnowflakeCatalog(
        username=catalog["username"],
        password=catalog["password"],
        account=catalog["accountName"],
        warehouse=warehouse,
        database=database,
        connection_params=connection_params,
        iceberg_volume=iceberg_volume,
    )


def create_tabular_catalog(catalog, data: CatalogData):
    """Create the tabular catalog from the catalog data.

    :param catalog: JSON catalog data loaded from secrets
    :param data: Optional data for catalog if provided will override data from secret store
    """
    warehouse = _get_warehouse(catalog, data.warehouse)
    iceberg_rest_url = _get_iceberg_rest_url(catalog, data.iceberg_rest_url)
    client_id, client_secret = _get_tabular_credentials(catalog)

    # Gets a user access token
    iceberg_rest_url = iceberg_rest_url.removesuffix("/")
    auth_url = iceberg_rest_url + ICEBERG_AUTH_URL_SUFFIX
    oauth_response = requests.post(
        auth_url,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    user_session_token = oauth_response.json()["access_token"]

    return bodosql.TabularCatalog(
        warehouse=warehouse,
        rest_uri=iceberg_rest_url,
        token=user_session_token,
    )


def create_glue_catalog(catalog, data: CatalogData):
    """Create the glue iceberg catalog from the catalog data.

    :param catalog: JSON catalog data loaded from secrets
    :param data: Optional data for catalog if provided will override data from secret store
    """
    warehouse = _get_warehouse(catalog, data.warehouse)
    return bodosql.GlueCatalog(warehouse=warehouse)


def _get_warehouse(catalog, warehouse) -> str:
    """Get catalog warehouse

    :param catalog: JSON catalog data loaded from secrets
    :param warehouse: Optional value, it will override value from catalog
    """
    warehouse = warehouse if warehouse else catalog.get("warehouse")
    if warehouse is not None:
        return warehouse

    raise ValueError(
        "No warehouse specified in either the catalog data or through the arguments."
    )


def _get_database(catalog, database) -> str:
    """Get catalog database

    :param catalog: JSON catalog data loaded from secrets
    :param database: Optional value, it will override value from catalog
    """
    database = database if database else catalog.get("database")
    if database is not None:
        return database

    raise ValueError(
        "No database specified in either the catalog data or through the arguments."
    )


def _get_iceberg_rest_url(catalog, iceberg_rest_url=None) -> str:
    """Get iceberg rest url
    Database name
    :param catalog: JSON catalog data loaded from secrets
    :param iceberg_rest_url: Optional value, it will override value from catalog
    :return: Iceberg Rest URL
    """
    iceberg_rest_url = (
        iceberg_rest_url if iceberg_rest_url else catalog.get("icebergRestUrl")
    )

    if iceberg_rest_url is not None:
        return iceberg_rest_url

    raise ValueError(
        "No icebergRestUrl specified in either the catalog data or through the arguments."
    )


def _get_tabular_credentials(catalog) -> tuple:
    """Parse credentials from catalog data

    :param catalog: JSON catalog data loaded from secrets
    :return: client_id, client_secret
    """
    data = catalog.get("credential")
    if data is None:
        raise ValueError("No credential specified in the catalog data.")

    credentials = data.split(":")
    if len(credentials) != 2:
        raise ValueError("Credential should be in the format 'client_id:client_secret'")

    return credentials[0], credentials[1]


# Users have to use the below helper functions to get the secrets from SSM.
# Calling AWS SSM APIs can be costly, especially for the Interactive SQL use case
# where it’ll be called every time the SQL cell is executed.
# To reduce this cost, we set up a simple in-memory TTL cache on this function.
@cached(
    cache=TTLCache(
        maxsize=256,
        ttl=3600,
    ),
    key=lambda name=None, _parallel=True: hashkey(name, _parallel),
)
def get_data(name=None, _parallel=True):
    """
     :param name: Name of the Catalog
     :param _parallel: Defaults to True
    :return: JSON object containing the Catalog data
    """
    if name is None:
        name = SNOWFLAKE_PC_CATALOG_NAME

    catalog_name = f"{CATALOG_PREFIX}-{name}"

    # Currently all the catalogs will be stored under default secret group.
    # Default secret group will be created at the time of workspace creation.
    return get(catalog_name, DEFAULT_SECRET_GROUP, _parallel)


def display_version():
    return "2.0.0"
