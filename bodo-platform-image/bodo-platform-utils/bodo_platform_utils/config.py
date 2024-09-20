import os
from bodo_platform_utils.types import CloudProvider

BODO_PLATFORM_WORKSPACE_REGION = os.environ.get("BODO_PLATFORM_WORKSPACE_REGION")
BODO_PLATFORM_WORKSPACE_UUID = os.environ.get("BODO_PLATFORM_WORKSPACE_UUID")
BODO_PLATFORM_CLOUD_PROVIDER = CloudProvider(
    os.environ.get("BODO_PLATFORM_CLOUD_PROVIDER")
)

BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES = os.environ.get(
    "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES", None
)
if BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES:
    parts = BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES.split("@")
    account_part = parts[1] if len(parts) > 1 else ""
    BODO_STORAGE_ACCOUNT_NAME = account_part.split(".")[0]
else:
    BODO_STORAGE_ACCOUNT_NAME = None

# This is related to Snowflake Partner Connect, we use the same constant in the backend
# to store the data coming from Partner Connect.
SNOWFLAKE_PC_CATALOG_NAME = "snowflake_pc"

CATALOG_PREFIX = "catalog-secret"
DEFAULT_SECRET_GROUP = "default"
