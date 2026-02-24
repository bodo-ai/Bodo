from __future__ import annotations

import os
from urllib.parse import urlparse

from pyarrow.fs import FileSystem
from pyiceberg.io import ADLS_ACCOUNT_KEY, ADLS_ACCOUNT_NAME
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.typedef import EMPTY_DICT, Properties

import bodo.io.utils


def _map_wasb_to_abfs(scheme: str, netloc: str) -> tuple[str, str]:
    """
    Map wasb and wasbs to abfss and abfs. Leaves others as is
    """
    if scheme == "wasb":
        scheme = "abfs"
        netloc = netloc.replace("blob.core.windows.net", "dfs.core.windows.net")
    elif scheme == "wasbs":
        scheme = "abfss"
        netloc = netloc.replace("blob.core.windows.net", "dfs.core.windows.net")
    elif scheme == "azure":
        return _map_wasb_to_abfs("abfs", netloc)

    return scheme, netloc


class BodoPyArrowFileIO(PyArrowFileIO):
    """
    A class that extends PyArrowFileIO to extend AzureFileSystem support.
    """

    @staticmethod
    def parse_location(
        location: str, *args, properties: Properties = EMPTY_DICT, **kwargs
    ) -> tuple[str, str, str]:
        """Return (scheme, netloc, path) for the given location.

        Uses DEFAULT_SCHEME and DEFAULT_NETLOC if scheme/netloc are missing.
        """

        uri = urlparse(location)
        if not uri.scheme or bodo.io.utils.is_windows_path(location):
            default_scheme = properties.get("DEFAULT_SCHEME", "file")
            default_netloc = properties.get("DEFAULT_NETLOC", "")
            return default_scheme, default_netloc, os.path.abspath(location)
        elif uri.scheme in ("hdfs", "viewfs"):
            return uri.scheme, uri.netloc, uri.path
        elif uri.scheme in ("abfs", "abfss", "azure", "wasbs", "wasb"):
            path = uri.path.removeprefix("/")
            if uri.username:
                path = f"{uri.username}/{path}"

            # Netloc is just host name, excluding any user-password
            netloc = uri.hostname
            assert netloc is not None

            # Map wasbs and wasb to abfss and abfs
            scheme, netloc = _map_wasb_to_abfs(uri.scheme, netloc)
            return scheme, netloc, path
        else:
            return uri.scheme, uri.netloc, f"{uri.netloc}{uri.path}"

    def _initialize_fs(self, scheme: str, netloc: str | None = None) -> FileSystem:
        if netloc:
            scheme, netloc = _map_wasb_to_abfs(scheme, netloc)

        if scheme in {"abfs", "abfss", "azure"}:
            account_name = None
            if netloc and netloc.endswith(".blob.core.windows.net"):
                account_name = netloc.removesuffix(".blob.core.windows.net")
            elif netloc and netloc.endswith(".dfs.core.windows.net"):
                account_name = netloc.removesuffix(".dfs.core.windows.net")
            elif netloc:
                pass
            else:
                account_name = self.properties.get(ADLS_ACCOUNT_NAME) or os.environ.get(
                    "AZURE_STORAGE_ACCOUNT_NAME"
                )
            self.properties[ADLS_ACCOUNT_NAME] = account_name

            account_key = self.properties.get(ADLS_ACCOUNT_KEY) or os.environ.get(
                "AZURE_STORAGE_ACCOUNT_KEY"
            )
            self.properties[ADLS_ACCOUNT_KEY] = account_key

        return super()._initialize_fs(scheme, netloc)
