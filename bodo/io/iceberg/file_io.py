import os
import typing as pt
from pathlib import PureWindowsPath
from urllib.parse import urlparse

from pyarrow.fs import FileSystem
from pyiceberg.io import ADLS_ACCOUNT_KEY, ADLS_ACCOUNT_NAME
from pyiceberg.io.pyarrow import PyArrowFileIO


def _map_wasb_to_abfs(scheme: str, netloc: str) -> tuple[str, str]:
    """
    Map wasb and wasbs to abfss and abfs. Leaves others as is
    """
    match scheme:
        case "wasb":
            scheme = "abfs"
            netloc = netloc.replace("blob.core.windows.net", "dfs.core.windows.net")
        case "wasbs":
            scheme = "abfss"
            netloc = netloc.replace("blob.core.windows.net", "dfs.core.windows.net")
        case _:
            pass
    return scheme, netloc


def _is_windows_path(path: str) -> bool:
    """
    Check if the given path is a Windows path (e.g. C:\\user\\data).
    """
    p = PureWindowsPath(path)

    # True if a typical Windows drive like "C:" or a UNC drive like "\\server\share"
    if p.drive:
        if len(p.drive) == 2 and p.drive[1] == ":":
            return True

        if p.drive.startswith("\\"):
            return True

    return False


class BodoPyArrowFileIO(PyArrowFileIO):
    """
    A class that extends PyArrowFileIO to extend AzureFileSystem support.
    """

    @staticmethod
    @pt.override
    def parse_location(location: str) -> tuple[str, str, str]:
        """Return the path without the scheme."""

        if _is_windows_path(location):
            return "file", "", os.path.abspath(location)

        uri = urlparse(location)
        if not uri.scheme:
            return "file", uri.netloc, os.path.abspath(location)
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

    @pt.override
    def _initialize_fs(self, scheme: str, netloc: str | None = None) -> FileSystem:
        if netloc:
            scheme, netloc = _map_wasb_to_abfs(scheme, netloc)

        if scheme in {"abfs", "abfss", "azure"}:
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

            account_key = self.properties.get(ADLS_ACCOUNT_KEY) or os.environ.get(
                "AZURE_STORAGE_ACCOUNT_KEY"
            )

            from pyarrow.fs import AzureFileSystem

            return AzureFileSystem(
                account_name=account_name,
                account_key=account_key,
            )

        return super()._initialize_fs(scheme, netloc)
