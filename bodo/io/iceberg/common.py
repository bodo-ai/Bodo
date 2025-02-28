"""
Common helper functions and types for Iceberg support.
"""

from __future__ import annotations

import importlib
import typing as pt
from urllib.parse import parse_qs, urlparse

import requests
from numba.core import types
from pyarrow.fs import FileSystem, FSSpecHandler

from bodo.utils.utils import BodoError, run_rank0

if pt.TYPE_CHECKING:  # pragma: no cover
    from pyiceberg.io import FileIO
    from pyiceberg.table import FileScanTask


class IcebergParquetInfo(pt.NamedTuple):
    """Named Tuple for Parquet info needed by Bodo"""

    # Iceberg file info
    file_task: FileScanTask
    # Iceberg Schema ID the parquet file was written with
    schema_id: int
    # Sanitized path to the parquet file for filesystem
    sanitized_path: str

    @property
    def path(self) -> str:
        return self.file_task.file.file_path

    @property
    def row_count(self) -> int:
        return self.file_task.file.record_count


def verify_pyiceberg_installed():
    """
    Verify that the PyIceberg package is installed.
    """

    try:
        return importlib.import_module("pyiceberg")
    except ImportError:
        raise BodoError(
            "Please install the pyiceberg package to use Iceberg functionality. "
            "You can install it by running 'pip install pyiceberg'."
        ) from None


T = pt.TypeVar("T")
TVals = T | tuple["TVals", ...]


def flatten_tuple(x: tuple[TVals, ...]) -> tuple[T]:
    """
    Flatten a tuple of tuples into a single tuple. This is needed
    to handle nested tuples in the schema group identifier due to
    nested data types.
    """
    values = []
    for val in x:
        if isinstance(val, tuple):
            values.extend(flatten_tuple(val))
        else:
            values.append(val)
    return tuple(values)


def flatten_concatenation(list_of_lists: list[list[pt.Any]]) -> list[pt.Any]:
    """
    Helper function to flatten a list of lists into a
    single list.

    Ref: https://realpython.com/python-flatten-list/

    Args:
        list_of_lists (list[list[Any]]): List to flatten.

    Returns:
        list[Any]: Flattened list.
    """
    flat_list: list[pt.Any] = []
    for row in list_of_lists:
        flat_list += row
    return flat_list


FieldID: pt.TypeAlias = int | tuple["FieldID", ...]
FieldIDs: pt.TypeAlias = tuple[FieldID, ...]
FieldName: pt.TypeAlias = str | tuple["FieldName", ...]
FieldNames: pt.TypeAlias = tuple[FieldName, ...]
SchemaGroupIdentifier: pt.TypeAlias = tuple[FieldIDs, FieldNames]


# ===================================================================
# Must match the values in bodo_iceberg_connector/schema_helper.py
# ===================================================================
# This is the key used for storing the Iceberg Field ID in the
# metadata of the Arrow fields.
# Taken from: https://github.com/apache/arrow/blob/c23a097965b5c626cbc91b229c76a6c13d36b4e8/cpp/src/parquet/arrow/schema.cc#L245.
ICEBERG_FIELD_ID_MD_KEY = "PARQUET:field_id"

# PyArrow stores the metadata keys and values as bytes, so we need
# to use this encoded version when trying to access existing
# metadata in fields.
b_ICEBERG_FIELD_ID_MD_KEY = str.encode(ICEBERG_FIELD_ID_MD_KEY)
# ===================================================================


@run_rank0
def get_rest_catalog_config(conn: str) -> tuple[str, str, str] | None:
    """
    Get the configuration for a rest catalog connection string.
    @param conn: Iceberg connection string.
    @return: Tuple of uri, user_token, warehouse if successful, None otherwise (e.g. invalid connection string or not a rest catalog).
    """
    parsed_conn = urlparse(conn)
    if parsed_conn.scheme.lower() not in {"http", "https"}:
        return None
    parsed_params = parse_qs(parsed_conn.query)
    # Clear the params
    parsed_conn = parsed_conn._replace(query="")
    uri = parsed_conn.geturl()

    user_token, credential, warehouse = (
        parsed_params.get("token"),
        parsed_params.get("credential"),
        parsed_params.get("warehouse"),
    )
    if user_token is not None:
        user_token = user_token[0]
    if warehouse is not None:
        warehouse = warehouse[0]
    # If we have a credential, we need to use it to get a user_token
    if credential is not None:
        credential = credential[0]
        client_id, client_secret = credential.split(":")
        user_token_request = requests.post(
            f"{uri}/v1/oauth/tokens",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if user_token_request.status_code != 200:
            raise BodoError(
                f"Unable to authenticate with {uri}. Please check your connection string."
            )
        user_token = user_token_request.json().get("access_token")

    if user_token is None:
        raise BodoError(
            f"Unable to authenticate with {uri}. Please check your connection string."
        )
    return uri, str(user_token), str(warehouse)


# ----------------------- Connection String Handling ----------------------- #


class IcebergConnectionType(types.Type):
    """
    Abstract base class for IcebergConnections
    """

    def __init__(self, name):  # pragma: no cover
        super().__init__(name=name)


def _fs_from_file_path(file_path: str, io: FileIO) -> FileSystem:
    """
    Construct a PyArrow FileSystem from a file path and a FileIO object.
    This is copied from pyiceberg.io.pyarrow._fs_from_file_path with
    a modification to use Bodo's changes to PyArrowFileIO in the monkey
    patch.
    """
    from pyiceberg.io.pyarrow import PyArrowFileIO

    # Bodo Change: Use the parse_location function from BodoPyArrowFileIO
    scheme, netloc, _ = PyArrowFileIO.parse_location(file_path)
    if isinstance(io, PyArrowFileIO):
        return io.fs_by_scheme(scheme, netloc)
    else:
        try:
            from pyiceberg.io.fsspec import FsspecFileIO

            if isinstance(io, FsspecFileIO):
                from pyarrow.fs import PyFileSystem

                return PyFileSystem(FSSpecHandler(io.get_fs(scheme)))
            else:
                raise ValueError(f"Expected PyArrowFileIO or FsspecFileIO, got: {io}")
        except ModuleNotFoundError as e:
            # When FsSpec is not installed
            raise ValueError(
                f"Expected PyArrowFileIO or FsspecFileIO, got: {io}"
            ) from e


def _format_data_loc(data_loc: str, fs: FileSystem) -> str:
    """
    Format the data location to be written to depending on the filesystem.
    """
    from pyarrow.fs import AzureFileSystem

    if isinstance(fs, AzureFileSystem) and data_loc.startswith("abfs"):
        # Azure filesystem only wants the container/path
        parsed = urlparse(data_loc)
        return f"{parsed.username}{parsed.path}"
    return data_loc
