"""
API used to translate Java BodoParquetInfo objects into
Python Objects usable inside Bodo.
"""

import os
from urllib.parse import urlparse


def standardize_path(path: str, warehouse_loc: str | None) -> str:
    if warehouse_loc is not None:
        warehouse_loc = warehouse_loc.replace("s3a://", "s3://").removeprefix("file:")

    if _has_uri_scheme(path):
        return (
            path.replace("s3a://", "s3://")
            .replace("wasbs://", "abfss://")
            .replace("wasb://", "abfs://")
            .replace("blob.core.windows.net", "dfs.core.windows.net")
            .removeprefix("file:")
        )
    elif warehouse_loc is not None:
        return os.path.join(warehouse_loc, path)
    else:
        return path


def _has_uri_scheme(path: str):
    """return True of path has a URI scheme, e.g. file://, s3://, etc."""
    try:
        return urlparse(path).scheme != ""
    except Exception:
        return False
