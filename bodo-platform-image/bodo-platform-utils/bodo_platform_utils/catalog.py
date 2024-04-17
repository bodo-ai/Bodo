from bodo_platform_utils.config import (
    CATALOG_PREFIX,
    SNOWFLAKE_PC_CATALOG_NAME,
    DEFAULT_SECRET_GROUP,
)
from bodo_platform_utils.secrets import get
from cachetools import TTLCache, cached
from cachetools.keys import hashkey


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
