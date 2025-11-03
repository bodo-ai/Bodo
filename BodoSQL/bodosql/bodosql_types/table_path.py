from __future__ import annotations

import json
from functools import cached_property
from urllib.parse import ParseResult, urlparse

import bodo
from bodo.io.parquet_pio import get_parquet_dataset


def check_tablepath_constant_arguments(
    file_path,
    file_type,
    conn_str,
    reorder_io,
    db_schema,
    bodo_read_as_dict,
    statistics_file,
):
    """
    Helper function used to do the majority of error checking for the TablePath
    Python API and JIT constructors. This handles entirely Python objects.
    """
    if not isinstance(file_path, str):
        raise ValueError("bodosql.TablePath(): Requires a 'file_path' string.")
    # Users must provide a file type
    if not isinstance(file_type, str):
        raise ValueError(
            "bodosql.TablePath(): Requires a 'file_type' string. File type(s) currently supported: (`parquet`, `sql`)"
        )
    if file_type not in ("pq", "sql"):
        raise ValueError(
            f"bodosql.TablePath(): `file_type` {file_type} not supported. File type(s) currently supported: (`parquet`, `sql`)"
        )
    # conn_str is required for sql
    if file_type == "sql":
        if conn_str is None:
            raise ValueError(
                "bodosql.TablePath(): `conn_str` is required for the `sql` `file_type`."
            )
        elif not isinstance(conn_str, str):
            raise ValueError("bodosql.TablePath(): `conn_str` must be a string")
        db_type, _ = bodo.ir.sql_ext.parse_dbtype(conn_str)
        if db_type == "iceberg":
            if db_schema is None:
                raise ValueError(
                    "bodosql.TablePath(): `db_schema` is required for iceberg database type."
                )
            elif not isinstance(db_schema, str):
                raise ValueError("bodosql.TablePath(): `db_schema` must be a string.")

    elif conn_str is not None:
        raise ValueError(
            "bodosql.TablePath(): `conn_str` is only supported for the `sql` `file_type`."
        )
    if not isinstance(reorder_io, bool):
        raise ValueError(
            "bodosql.TablePath(): `reorder_io` must be a boolean if provided."
        )

    if not (
        (bodo_read_as_dict is None)
        or (
            isinstance(bodo_read_as_dict, list)
            and all(isinstance(item, str) for item in bodo_read_as_dict)
        )
    ):
        raise ValueError(
            "bodosql.TablePath(): `bodo_read_as_dict` must be a constant list of strings if provided."
        )

    if not ((statistics_file is None) or (isinstance(statistics_file, str))):
        raise ValueError(
            "bodosql.TablePath(): `statistics_file` must be a constant string if provided."
        )


def convert_tablepath_constructor_args(
    file_path, file_type, conn_str, reorder_io, bodo_read_as_dict, statistics_file
):
    """
    Helper function to modify the TablePath arguments in a consistent way across
    JIT code and the Python API. This takes entirely Python objects.
    """
    # Accept file types as case insensitive
    file_type = file_type.strip().lower()
    # Always store parquet in the shortened form.
    if file_type == "parquet":
        file_type = "pq"
    if reorder_io is None:
        # TODO: Modify the default?
        # Perhaps SQL should be False because you need to access
        # a running DB but pq should be True?
        reorder_io = True

    return (
        file_path,
        file_type,
        conn_str,
        reorder_io,
        bodo_read_as_dict,
        statistics_file,
    )


def _get_path_protocol(path: str) -> str:
    """
    Get protocol of a path (e.g. s3://, or "" for file).
    """
    if bodo.io.iceberg.file_io._is_windows_path(path):
        return ""

    parsed_url: ParseResult = urlparse(path)
    return parsed_url.scheme


def load_statistics(
    statistics_file: str,
) -> tuple[int | None, dict[str, int]]:
    """
    Load table statistics from a file.

    Supported keys are:
    - row_count (int)
    - ndv (dict)

    Args:
        statistics_file (str): Path to the statistics file.
            This must be a file on the local filesystem.

    Returns:
        tuple[Optional[int], dict[str, int]]:
            - Row count if provided, else None
            - Map of column names to their NDV estimates. This may
              have estimates for only some of the columns.
    """

    protocol = _get_path_protocol(statistics_file)
    statistics_file = statistics_file.rstrip("/")

    # TODO Add support for loading the statistics from a S3/ADLS/HTTPS URI.
    if protocol != "":
        raise ValueError(
            f"Unsupported protocol '{protocol}' for the statistics file ('{statistics_file}')."
        )

    with open(statistics_file) as f:
        stats: dict = json.load(f)

    row_count: int | None = stats.get("row_count", None)
    ndv: dict[str, int] = stats.get("ndv", {})

    if not all((isinstance(k, str) and isinstance(v, int)) for k, v in ndv.items()):
        raise ValueError(
            f"'ndv' field in the statistics file ('{statistics_file}') must have string keys and integer values!"
        )

    return row_count, ndv


class TablePath:
    """
    Python class used to hold information about an individual table
    that should be loaded from a file. The file_path is a string
    that should describe the type of file to read.
    """

    def __init__(
        self,
        file_path: str,
        file_type: str,
        *,
        conn_str: str | None = None,
        reorder_io: bool | None = None,
        db_schema: str | None = None,
        bodo_read_as_dict: list[str] | None = None,
        statistics_file: str | None = None,
    ):
        # Update the arguments.
        (
            file_path,
            file_type,
            conn_str,
            reorder_io,
            bodo_read_as_dict,
            statistics_file,
        ) = convert_tablepath_constructor_args(
            file_path,
            file_type,
            conn_str,
            reorder_io,
            bodo_read_as_dict,
            statistics_file,
        )

        # Check the arguments
        check_tablepath_constant_arguments(
            file_path,
            file_type,
            conn_str,
            reorder_io,
            db_schema,
            bodo_read_as_dict,
            statistics_file,
        )

        self._file_path = file_path
        self._file_type = file_type
        self._conn_str = conn_str
        self._reorder_io = reorder_io
        self._db_schema = db_schema
        self._bodo_read_as_dict = bodo_read_as_dict
        # 'row_count' is the number of rows in the table or None if not known.
        # 'ndv' is a dictionary mapping the column name to its distinct count.
        # This may only have NDVs for some of the columns.
        self._statistics = {"row_count": None, "ndv": {}}

        # Load the statistics from the statistics_file if one is provided:
        if statistics_file is not None:
            row_count, ndv = load_statistics(statistics_file)
            self._statistics["row_count"] = row_count
            self._statistics["ndv"] = ndv

    def __key(self):
        bodo_read_dict = (
            None if self._bodo_read_as_dict is None else tuple(self._bodo_read_as_dict)
        )
        return (
            self._file_path,
            self._file_type,
            self._conn_str,
            self._reorder_io,
            self._db_schema,
            bodo_read_dict,
            self.statistics_json_str,
        )

    @cached_property
    def statistics_json_str(self):
        return json.dumps(self._statistics)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        """
        Overload equality operator. This is done to enable == in testing
        """
        if isinstance(other, TablePath):
            return self.__key() == other.__key()
        return False

    def equals(self, other):
        """
        Equivalent to ==. Done to ensure DataFrame and TablePath
        can use the same API for equality when testing a BodoSQLContext.
        """
        return self == other

    def __repr__(self):
        return f"TablePath({self._file_path!r}, {self._file_type!r}, conn_str={self._conn_str!r}, reorder_io={self._reorder_io!r}), db_schema={self._db_schema!r}, bodo_read_as_dict={self._bodo_read_as_dict!r}, statistics={self.statistics_json_str!r}"

    def __str__(self):
        return f"TablePath({self._file_path}, {self._file_type}, conn_str={self._conn_str}, reorder_io={self._reorder_io}), db_schema={self._db_schema}, bodo_read_as_dict={self._bodo_read_as_dict}, statistics={self.statistics_json_str}"

    @cached_property
    def estimated_row_count(self) -> int | None:
        if self._statistics["row_count"] is not None:
            # If available, use the row count provided in the statistics file.
            return self._statistics["row_count"]
        elif self._file_type == "pq":
            return get_parquet_dataset(self._file_path)._bodo_total_rows
        else:
            return None
