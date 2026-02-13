from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import traceback
import warnings
from enum import Enum
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import pyarrow as pa
from mpi4py import MPI
from numba.core import types

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.timestamptz_ext import ArrowTimestampTZType
from bodo.io.helpers import (
    _get_numba_typ_from_pa_typ,
    update_env_vars,
)
from bodo.libs.dict_arr_ext import dict_str_arr_type
from bodo.libs.map_arr_ext import contains_map_array
from bodo.utils import tracing
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.typing import BodoError, BodoWarning, is_str_arr_type, raise_bodo_error

# Imports for typechecking
if TYPE_CHECKING:  # pragma: no cover
    from snowflake.connector import SnowflakeConnection
    from snowflake.connector.cursor import ResultMetadata, SnowflakeCursor
    from snowflake.connector.result_batch import (
        ArrowResultBatch,
        JSONResultBatch,
        ResultBatch,
    )

# How long the schema / typeof probe query should run for in the worst case.
# This is to guard against increasing compilation time prohibitively in case there are
# issues with Snowflake, the data, etc.
SF_READ_SCHEMA_PROBE_TIMEOUT = 20

# Whether to do a probe query to determine whether string columns should be
# dictionary-encoded. This doesn't effect the _bodo_read_as_dict argument.
SF_READ_AUTO_DICT_ENCODE_ENABLED = True

# A configurable variable by which we determine whether to dictionary-encode
# a string column.
# Encode if num of unique elem / num of total rows <= SF_READ_DICT_ENCODE_CRITERION
SF_READ_DICT_ENCODE_CRITERION = 0.5


# Limits the dictionary size when deciding dictionary-encoded columns for the streaming
# case since large dictionaries make streaming write extremely slow (also join slow).
# Currently set to batch size to avoid dictionaries being larger than a batch to limit
# overheads as a heuristic.
# More details here: https://bodo.atlassian.net/browse/BSE-1200
# TODO[BSE-1200] revisit the limit when join issues are resolved.
SF_STREAM_READ_DICT_ENCODE_LIMIT = bodo.bodosql_streaming_batch_size

# How long the dictionary encoding probe query should run for in the worst case.
# This is to guard against increasing compilation time prohibitively in case there are
# issues with Snowflake, the data, etc.
SF_READ_DICT_ENCODING_PROBE_TIMEOUT = 10

# Default behavior if the query to determine dictionary encoding times out.
# This is false by default since dict encoding is an optimization, and in cases where
# we cannot definitively determine if it should be used, we should not use it. The
# config flag is useful in cases where we (developers) want to test certain situations
# manually.
SF_READ_DICT_ENCODING_IF_TIMEOUT = False

# The maximum number of rows a table can contain to be defined
# as a small table. This is used to determine whether to use
# dictionary encoding by default for all string columns. The
# justification for this is that small tables are either small,
# so the additional overhead of dictionary encoding is negligible,
# or their columns will be increased in size via a JOIN operation,
# so dictionary encoding will then be necessary as the values will
# be repeated.
SF_SMALL_TABLE_THRESHOLD = 100_000

# Maximum number of rows to read from Snowflake in the probe query
# This is calculated as # of string columns * # of rows
# The default 100M should take a negligible amount of time.
# This default value is based on empirical benchmarking to have
# a good balance between accuracy of the query and compilation
# time. Find more detailed analysis and results here:
# https://bodo.atlassian.net/wiki/spaces/B/pages/1134985217/Support+reading+dictionary+encoded+string+columns+from+Snowflake#Prediction-query-and-heuristic
SF_READ_DICT_ENCODING_PROBE_ROW_LIMIT = 100_000_000


class UnknownSnowflakeType(Enum):
    VARIANT = "variant"
    OBJECT = "object"
    LIST = "list"


SCALE_TO_UNIT_PRECISION: dict[int, Literal["s", "ms", "us", "ns"]] = {
    0: "s",
    3: "ms",
    6: "us",
    9: "ns",
}
INT_BITSIZE_TO_ARROW_DATATYPE = {
    1: pa.int8(),
    2: pa.int16(),
    4: pa.int32(),
    8: pa.int64(),
    16: pa.decimal128(38, 0),
}


def type_code_to_arrow_type(
    code: int, m: ResultMetadata, tz: str, is_select_q: bool
) -> pa.DataType | UnknownSnowflakeType:
    """
    Mapping of the Snowflake field types. Most are taken from the Snowflake Connector
    except for the following:
        - 0:  Number / Int: Bodo starts with Decimal and does further type inference
        - 6, 7, 8: Timestamps: Uses scale from metadata for typing
        - 9:  Object: Snowflake types output as string, Bodo derives fixed Struct or Map (NOT IMPLEMENTED YET)
        - 10: Array: Snowflake types output as string, Bodo derives fixed Array typing
        - 12: Time: Similar to Timestamp

    https://github.com/snowflakedb/snowflake-connector-python/blob/dcf10e8c7ce13a5288104b28329d3c9e8ffffc5a/src/snowflake/connector/constants.py#L35
    https://docs.snowflake.com/en/user-guide/python-connector-api.html#label-python-connector-type-codes
    """

    # Number / Int - Always Signed
    if code == 0:
        assert m.precision is not None
        assert m.scale is not None
        return pa.decimal128(m.precision, m.scale) if is_select_q else pa.int64()
    # Floating-Point - Always 64 bits / Double
    elif code == 1:
        return pa.float64()
    # String
    elif code == 2:
        return pa.string()
    # Dates - Snowflake stores in days (aka 32-bit)
    elif code == 3:
        return pa.date32()
    # Timestamp - Seems to be unused?
    elif code == 4:
        return pa.time64("ns")
    # Variant / Union Type
    elif code == 5:
        return UnknownSnowflakeType.VARIANT
    # Timestamp stored in UTC - TIMESTAMP_LTZ
    elif code == 6:
        assert m.scale is not None
        return pa.timestamp(SCALE_TO_UNIT_PRECISION[m.scale], tz=tz)
    # Timestamp with a timezone offset per item - TIMESTAMP_TZ
    elif code == 7:
        assert m.scale is not None
        return ArrowTimestampTZType()
    # Timestamp without a timezone - TIMESTAMP_NTZ
    elif code == 8:
        assert m.scale is not None
        return pa.timestamp(SCALE_TO_UNIT_PRECISION[m.scale])
    # Object -> Map / Struct - Connector doesn't support pa.struct
    elif code == 9:
        return UnknownSnowflakeType.OBJECT
    # List - Connector doesn't support pa.list
    elif code == 10:
        return UnknownSnowflakeType.LIST
    # Binary
    elif code == 11:
        return pa.binary()
    # Time
    elif code == 12:
        assert m.scale is not None
        return (
            {
                0: pa.time32("s"),
                3: pa.time32("ms"),
                6: pa.time64("us"),
                9: pa.time64("ns"),
            }
        )[m.scale]
    # Boolean
    elif code == 13:
        return pa.bool_()
    # Geographic - No Core Arrow Equivalent
    elif code == 14:
        return pa.string()
    else:
        raise BodoError(f"Unknown Snowflake Type Code: {code}")


def _import_snowflake_connector_logging() -> None:  # pragma: no cover
    """
    Helper function to set logging after
    importing snowflake.connector.
    """
    if int(os.environ.get("BODO_SF_DEBUG_LEVEL", "0")) >= 2:
        for logger_name in ("snowflake.connector",):
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(threadName)s %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(ch)


def gen_snowflake_schema(
    column_names, column_datatypes, column_precisions=None
):  # pragma: no cover
    """Generate a dictionary where column is key and
    its corresponding bodo->snowflake datatypes is value

    Args:
        column_names (array-like): Array of DataFrame column names
        column_datatypes (array-like): Array of DataFrame column datatypes
        column_precisions (array-like, optional): Array of precision values for each column.
        The values are only meaningful for string and timestamp columns. A value of -1 means that the
        precision is unknown so the maximum should be used.

    Returns:
        sf_schema (dict): {col_name : snowflake_datatype}
    Raises BodoError for unsupported datatypes when writing to snowflake.
    """

    def get_precision(col_idx: int) -> int:
        """Helper to get the precision for the current column - note that this
        helper only covers the common case - some types have more specific
        rules"""
        if column_precisions is None:
            precision = 9
        else:
            precision = column_precisions[col_idx]
            precision = 9 if precision == -1 else precision
        return precision

    sf_schema = {}
    for col_idx, (col_name, col_type) in enumerate(zip(column_names, column_datatypes)):
        if col_name == "":
            raise BodoError("Column name cannot be empty when writing to Snowflake.")
        # [BE-3587] need specific tz for each column type.
        if isinstance(col_type, bodo.types.DatetimeArrayType):
            precision = get_precision(col_idx)
            if col_type.tz is not None:
                sf_schema[col_name] = f"TIMESTAMP_LTZ({precision})"
            else:
                sf_schema[col_name] = f"TIMESTAMP_NTZ({precision})"
        elif col_type == bodo.types.timestamptz_array_type:
            precision = get_precision(col_idx)
            sf_schema[col_name] = f"TIMESTAMP_TZ({precision})"
        elif col_type == bodo.types.datetime_datetime_type:
            precision = get_precision(col_idx)
            sf_schema[col_name] = f"TIMESTAMP_NTZ({precision})"
        elif col_type == bodo.types.datetime_date_array_type:
            sf_schema[col_name] = "DATE"
        elif isinstance(col_type, bodo.types.TimeArrayType):
            # Note: The actual result may not match the precision
            # https://community.snowflake.com/s/article/Nano-second-precision-lost-after-Parquet-file-Unload
            if column_precisions is None:
                if col_type.precision in [0, 3, 6]:
                    precision = col_type.precision
                elif col_type.precision == 9:
                    # TODO(njriasan): Remove this branch eventually?
                    # Set precision to 6 due to snowflake limitation
                    # https://community.snowflake.com/s/article/Nano-second-precision-lost-after-Parquet-file-Unload
                    if bodo.get_rank() == 0:
                        warnings.warn(
                            BodoWarning(
                                f"to_sql(): {col_name} time precision will be lost.\nSnowflake loses nano second precision when exporting parquet file using COPY INTO.\n"
                                " This is due to a limitation on Parquet V1 that is currently being used in Snowflake"
                            )
                        )
                    precision = 6
                else:
                    raise ValueError("Unsupported Precision Found in Bodo Time Array")
            else:
                precision = get_precision(col_idx)
            sf_schema[col_name] = f"TIME({precision})"
        elif isinstance(col_type, types.Array):
            numpy_type = col_type.dtype.name
            if numpy_type.startswith("datetime"):
                precision = get_precision(col_idx)
                sf_schema[col_name] = f"TIMESTAMP_NTZ({precision})"
            # NOTE: Bodo matches Pandas behavior
            # and prints same warning and save it as a number.
            elif numpy_type.startswith("timedelta"):
                sf_schema[col_name] = "NUMBER(38, 0)"
                if bodo.get_rank() == 0:
                    warnings.warn(
                        BodoWarning(
                            f"to_sql(): {col_name} with type 'timedelta' will be written as integer values (ns frequency) to the database."
                        )
                    )
            # TODO: differentiate unsigned, int8, int16, ...
            elif numpy_type.startswith(("int", "uint")):
                sf_schema[col_name] = "NUMBER(38, 0)"
            elif numpy_type.startswith("float"):
                sf_schema[col_name] = "REAL"
        elif col_type == bodo.types.timedelta_array_type:
            sf_schema[col_name] = "NUMBER(38, 0)"
            if bodo.get_rank() == 0:
                warnings.warn(
                    BodoWarning(
                        f"to_sql(): {col_name} with type 'timedelta' will be written as integer values (ns frequency) to the database."
                    )
                )
        elif is_str_arr_type(col_type):
            if column_precisions is None or column_precisions[col_idx] < 0:
                sf_schema[col_name] = "TEXT"
            else:
                sf_schema[col_name] = f"VARCHAR({column_precisions[col_idx]})"
        elif col_type == bodo.types.binary_array_type:
            sf_schema[col_name] = "BINARY"
        elif col_type == bodo.types.boolean_array_type:
            sf_schema[col_name] = "BOOLEAN"
        # TODO: differentiate between unsigned vs. signed, 8, 16, 32, 64
        elif isinstance(col_type, bodo.types.IntegerArrayType):
            sf_schema[col_name] = "NUMBER(38, 0)"
        elif isinstance(col_type, bodo.types.FloatingArrayType):
            sf_schema[col_name] = "REAL"
        elif isinstance(col_type, bodo.types.DecimalArrayType):
            # TODO(njriasan): Integrate column_precisions when we have accurate
            # information from BodoSQL.
            sf_schema[col_name] = f"NUMBER({col_type.precision}, {col_type.scale})"
        elif isinstance(
            col_type,
            bodo.types.ArrayItemArrayType,
        ):
            if contains_map_array(col_type):
                raise_bodo_error("Nested MapArrayType is not supported.")
            sf_schema[col_name] = "ARRAY"

        elif isinstance(
            col_type,
            bodo.types.StructArrayType,
        ):
            if contains_map_array(col_type):
                raise_bodo_error("Nested MapArrayType is not supported.")
            sf_schema[col_name] = "OBJECT"

        elif isinstance(col_type, bodo.types.MapArrayType):
            if (
                not col_type.key_arr_type == bodo.types.string_array_type
                and bodo.get_rank() == 0
            ):
                warning = BodoWarning(
                    f"Snowflake does not support objects with non-string key type {col_type.key_arr_type}. Column {col_name} will be parsed as {{'key': key_value, 'value': value_value }}."
                )
                warnings.warn(warning)
            if contains_map_array(col_type.value_arr_type):
                raise_bodo_error("Nested MapArrayType is not supported.")
            sf_schema[col_name] = "OBJECT"
        # See https://bodo.atlassian.net/browse/BSE-1525
        elif col_type == bodo.types.null_array_type:
            sf_schema[col_name] = "VARCHAR"
        else:
            raise BodoError(
                f"Conversion from Bodo array type {col_type} to snowflake type for {col_name} not supported yet."
            )

    return sf_schema


# SF_WRITE_COPY_INTO_ON_ERROR (str):
# Action to take when `COPY INTO` statements fail.
#  -  "continue": Continue to load the file if errors are found.
#  -  "skip_file": Skip a file when an error is found.
#  -  "skip_file_<num>": Skip a file when the number of error rows
#         found in the file is equal to or exceeds the specified number.
#  -  "skip_file_<num>%": Skip a file when the percentage of error rows
#         found in the file exceeds the specified percentage.
#  -  "abort_statement": Abort the load operation if any error is
#         found in a data file.
# Default follows documentation for Snowflake's COPY INTO command:
# (https://docs.snowflake.com/en/sql-reference/sql/copy-into-table.html#copy-options-copyoptions)
SF_WRITE_COPY_INTO_ON_ERROR = "abort_statement"

# SF_WRITE_PARQUET_CHUNK_SIZE (int):
# Chunk size to use when writing dataframe to Parquet files, measured by
# the uncompressed memory usage of the dataframe to be compressed (in bytes).
# Decreasing the chunksize allows more load operations to run in parallel
# during `COPY_INTO`, while increasing the chunksize allows less processing
# overhead for each Parquet file. See Snowflake's File Sizing Best Practices
# and Limitations for guidance on how to choose this value:
# (https://docs.snowflake.com/en/user-guide/data-load-considerations-prepare.html#file-sizing-best-practices-and-limitations)
SF_WRITE_PARQUET_CHUNK_SIZE = int(256e6)

# SF_WRITE_PARQUET_COMPRESSION (str):
# The compression algorithm to use for Parquet files uploaded to Snowflake
# internal stage. Can be any compression algorithm supported by Pyarrow, but
# "snappy" and "gzip " should work best as they are specifically suited for parquet:
# "NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD". See this link for
# supported codecs: https://github.com/apache/parquet-format/blob/master/Compression.md
SF_WRITE_PARQUET_COMPRESSION = "snappy"

# SF_WRITE_STREAMING_NUM_FILES (int):
# For streaming COPY INTO, number of files to include in each separate call
# to `COPY INTO`, summed across all ranks. A lower value means COPY INTO will
# run more frequently with fewer files in each call, allowing more parallelism
# and perhaps incurring more overhead.
SF_WRITE_STREAMING_NUM_FILES = 128

# SF_WRITE_ASYNC_QUERY_FREQ (float):
# For asynchronous Snowflake SQL queries, how often to query the Snowflake
# cursor for the results of the asynchronous query. Snowflake internally
# uses 1 second in their user docs:
# https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-example#checking-the-status-of-a-query
SF_WRITE_ASYNC_QUERY_FREQ = 0.5

# SF_WRITE_UPLOAD_USING_PUT (bool):
# If True, `to_sql` saves the dataframe to Parquet files in a local
#     temporary directory, then uses Snowflake's PUT command to upload
#     to internal stage (https://docs.snowflake.com/en/sql-reference/sql/put.html)
#     This method supports all Snowflake stage types but may be slower.
# If False, `to_sql` directly uploads Parquet files to S3/ADLS/GCS using Bodo's
#     internal filesystem-write infrastructure. This method is faster but does
#     not support Azure and GCS-backed Snowflake accounts.
SF_WRITE_UPLOAD_USING_PUT = False


def execute_query(
    cursor: SnowflakeCursor,
    query: str,
    timeout: int | None,
) -> SnowflakeCursor | None:  # pragma: no cover
    """
    Execute a Snowflake Query with Special Timeout Handling
    This function executes independently of ranks

    Returns:
        None if the query timed out, otherwise the resulting cursor

    Raises:
        Any other snowflake.connector.errors.ProgrammingError
        unrelated to timeouts
    """
    try:
        return cursor.execute(query, timeout=timeout)
    except snowflake.connector.errors.ProgrammingError as e:
        # Catch timeouts
        if "SQL execution canceled" in str(e):
            return None
        else:
            raise e


def escape_col_name(col_name: str) -> str:
    """Helper Function to Escape Snowflake Column Names"""
    return '"{}"'.format(col_name.replace('"', '""'))


def matches_unquoted_id_rules(col_name: str) -> bool:
    """
    Check if col_name follows Snowflake's unquoted identifier rules:
    https://docs.snowflake.com/en/sql-reference/identifiers-syntax
    """
    return all(c.isalnum() or c == "_" or c == "$" for c in col_name) and (
        (len(col_name) == 0) or col_name[0] != "$"
    )


def snowflake_connect(
    conn_str: str, is_parallel: bool = False
) -> SnowflakeConnection:  # pragma: no cover
    """
    From Snowflake connection URL, connect to Snowflake.

    Args:
        conn_str: Snowflake Connection URL. See parse_snowflake_conn_str for specific format
        is_parallel: True if this function being is called from all
            ranks, and False otherwise

    Returns
        conn: Snowflake Connection object
    """
    ev = tracing.Event("snowflake_connect", is_parallel=is_parallel)
    params = bodo.io.utils.parse_snowflake_conn_str(conn_str)

    # Set a short login timeout so people don't have to wait the default
    # 60 seconds to find out they added the wrong credentials.
    params["login_timeout"] = 5
    # Bodo executes async queries to perform writes. While most should be quick,
    # some could take longer than the Snowflake default of 5 minutes so we need to
    # ensure ABORT_DETACHED_QUERY is set to False. This should be the case typically,
    # but some organizations may have updated their value.
    if "session_parameters" not in params:
        params["session_parameters"] = {}
    if params["session_parameters"].get("ABORT_DETACHED_QUERY", False):
        warning = BodoWarning(
            "Session parameter 'ABORT_DETACHED_QUERY' found in connection string and will be ignored. "
            "Bodo forces this value to always be False because it may submit async queries."
        )
        warnings.warn(warning)
    params["session_parameters"]["ABORT_DETACHED_QUERY"] = False

    # When running benchmarks, we want to ensure that Snowflake is not returning
    # results directly from its result cache (i.e. retrieval optimization).
    # (Ref: https://docs.snowflake.com/en/user-guide/querying-persisted-results#retrieval-optimization)
    # Setting the 'USE_CACHED_RESULT' session parameter
    # (https://docs.snowflake.com/en/sql-reference/parameters#use-cached-result)
    # forces Snowflake to skip the result cache and execute the query even if
    # it has a cached result. Note that this only affects the result cache and not
    # the data cache on its warehouse.
    # (Ref: https://community.snowflake.com/s/article/Caching-in-the-Snowflake-Cloud-Data-Platform)
    if os.environ.get("BODO_DISABLE_SF_RESULT_CACHE", "0") == "1":
        if params["session_parameters"].get("USE_CACHED_RESULT", False):
            warning = BodoWarning(
                "Session parameter 'USE_CACHED_RESULT' found in connection string and "
                "will be ignored since BODO_DISABLE_SF_RESULT_CACHE is set to 1."
            )
            warnings.warn(warning)
        params["session_parameters"]["USE_CACHED_RESULT"] = False

    try:
        import snowflake.connector
    except ImportError:
        raise BodoError(
            "Snowflake Python connector packages not found. "
            "Using 'to_sql' with Snowflake requires snowflake-connector-python. "
            "This can be installed by calling 'conda install -c conda-forge snowflake-connector-python' "
            "or 'pip install snowflake-connector-python'."
        )
    # If bodo_use_decimal is enabled, reads NUMBER columns as decimals instead of float64
    params["arrow_number_to_decimal"] = bodo.bodo_use_decimal
    conn = snowflake.connector.connect(**params)
    platform_region_str = os.environ.get("BODO_PLATFORM_WORKSPACE_REGION", None)
    if platform_region_str and bodo.get_rank() == 0:
        # Normalize to all lower case
        platform_region_str = platform_region_str.lower()
        platform_cloud_provider = os.environ.get("BODO_PLATFORM_CLOUD_PROVIDER", None)
        if platform_cloud_provider is not None:
            platform_cloud_provider = platform_cloud_provider.lower()
        cur = conn.cursor()
        # This query is very fast, taking at most .1 seconds in testing including
        # loading the data.
        cur.execute("select current_region()")
        arrow_data: pa.Table = cur.fetch_arrow_all()  # type: ignore
        sf_region_str = arrow_data[0][0].as_py()
        cur.close()
        # Parse the snowflake output
        region_parts = sf_region_str.split("_")
        # AWS and Azure use - instead of _. Otherwise all
        # of the region strings should match once we normalize
        # to all lower case. Snowflake also appends the cloud provider
        # to the front of the output.
        # https://docs.snowflake.com/en/user-guide/admin-account-identifier.html#region-ids
        sf_cloud_provider = region_parts[0].lower()
        sf_cloud_region = "-".join(region_parts[1:]).lower()
        if platform_cloud_provider and platform_cloud_provider != sf_cloud_provider:
            warning = BodoWarning(
                "Performance Warning: The Snowflake warehouse and Bodo platform are on different cloud providers. "
                + f"The Snowflake warehouse is located on {sf_cloud_provider}, but the Bodo cluster is located on {platform_cloud_provider}. "
                + "For best performance we recommend using your cluster and Snowflake account in the same region with the same cloud provider."
            )
            warnings.warn(warning)
        elif platform_region_str != sf_cloud_region:
            warning = BodoWarning(
                "Performance Warning: The Snowflake warehouse and Bodo platform are in different cloud regions. "
                + f"The Snowflake warehouse is located in {sf_cloud_region}, but the Bodo cluster is located in {platform_region_str}. "
                + "For best performance we recommend using your cluster and Snowflake account in the same region with the same cloud provider."
            )
            warnings.warn(warning)
    ev.finalize()
    return conn


def _try_convert_decimal_type_to_integer_type(type: pa.Decimal128Type) -> pa.DataType:
    """Try to convert a Decimal type to an integer representation
    without any loss of precision based on its static precision
    and scale information.

    Args:
        type (pa.DataType): The type to convert.

    Returns:
        pa.DataType: An equivalent integer type if conversion is possible
        or the original type if its not possible.
    """
    # Note: We add a defensive isinstance check that shouldn't be necessary.
    if isinstance(type, pa.Decimal128Type) and type.scale == 0 and type.precision < 19:
        # The type fits in an integer
        return INT_BITSIZE_TO_ARROW_DATATYPE[precision_to_byte_size(type.precision)]
    else:
        return type


def precision_to_byte_size(precision: int) -> int:
    """
    Helper function that maps the precision value of a fixed point number
    to the number of bytes needed to store it.

    returns -1 if the precision cannot be supported without loss of precision
    """
    if precision < 3:
        return 1
    elif precision < 5:
        return 2
    elif precision < 10:
        return 4
    elif precision < 19:
        return 8
    else:
        return -1


def precision_to_numpy_dtype(precision: int) -> int:
    """
    Helper function that maps the precision value of a fixed point number
    to a numpy dtype that can store it.

    returns None if the precision cannot be supported without loss of precision
    """
    byte_size = precision_to_byte_size(precision)
    if byte_size == 1:
        return types.int8
    elif byte_size == 2:
        return types.int16
    elif byte_size == 4:
        return types.int32
    elif byte_size == 8:
        return types.int64


def get_number_types_from_metadata(
    cursor: SnowflakeCursor,
    sql_query: str,
    orig_table: str | None,
    orig_table_indices: tuple[int, ...] | None,
    downcast_decimal_to_double: bool,
    cols_to_check: list[tuple[int, str, pa.Decimal128Type | pa.Decimal256Type]],
):
    """
    Determine the smallest possible integer size for all NUMBER columns
    in the output of a Snowflake read. This uses SYSTEM$TYPEOF for Snowflake
    internal sizing.

    Args:
        cursor: Snowflake cursor to submit metadata queries to
        sql_query: Source table or query to get output typing for
        orig_table: Original table name, to be used if sql_query is not
            a table name. If provided, must guarantee that the sql_query only performs
            a selection of a subset of the table's columns, and does not rename
            any of the columns from the input table. Defaults to None.
        orig_table_indices: The indices for each column
            in the original table. This is to handle renaming and replace name based reads with
            index based reads.
        downcast_decimal_to_double: Force that any remaining decimal columns are typed to float64
        cols_to_check: List of (int, str, DecimalType) representing the idx, name, and type of every
            NUMBER column

    Returns:
        - List of (int, str, DataType) representing column idx, name, and final type
            If metadata probing times out, just reuses the input cols_to_check
        - Columns that we couldn't get metadata for due to timeout. None if no columns
    """

    col_idxs_to_check = [c[0] for c in cols_to_check]
    col_names_to_check = [c[1] for c in cols_to_check]

    schema_probe_query = (
        "SELECT "
        + ", ".join(f"SYSTEM$TYPEOF({escape_col_name(x)})" for x in col_names_to_check)
        + f" FROM ({sql_query}) LIMIT 1"
    )

    probe_res = execute_query(
        cursor,
        schema_probe_query,
        timeout=SF_READ_SCHEMA_PROBE_TIMEOUT,
    )

    typing_table: pa.Table | None = None

    # Retry if first query failed and original table context is known
    if (
        (probe_res is None or (typing_table := probe_res.fetch_arrow_all()) is None)
        and orig_table is not None
        and orig_table_indices is not None
    ):
        schema_probe_query = (
            "SELECT "
            + ", ".join(
                # Note: Snowflake/SQL is 1 indexed
                f"SYSTEM$TYPEOF(${orig_table_indices[i] + 1})"
                for i in col_idxs_to_check
            )
            + f" FROM {orig_table} LIMIT 1"
        )
        probe_res = execute_query(
            cursor,
            schema_probe_query,
            timeout=SF_READ_SCHEMA_PROBE_TIMEOUT,
        )

    if probe_res is None or (typing_table := probe_res.fetch_arrow_all()) is None:
        return (cols_to_check, col_names_to_check)

    new_col_info: list[tuple[int, str, pa.DataType]] = []

    # Note, this assumes that the output metadata columns are in the
    # same order as the columns we checked in the probe query
    for i, (_, typing_info) in enumerate(typing_table.to_pylist()[0].items()):
        idx, name, dtype = cols_to_check[i]

        # Parse output NUMBER(__,_)[SBx] to get the byte width x
        number_regex = re.search(r"NUMBER\(\d+,\d+\)\[SB(\d+)\]", typing_info)
        assert number_regex is not None
        byte_size = int(number_regex.group(1))

        # Map Byte Width for Integer Only Columns
        if dtype.scale == 0:
            if byte_size <= 8:
                out_dtype = INT_BITSIZE_TO_ARROW_DATATYPE[byte_size]
            else:
                # Maintain the precision from Snowflake
                out_dtype = dtype
        # Any non-16 byte decimal columns map to double, unless
        # we force them to be decimals.
        elif (
            byte_size <= 8 or downcast_decimal_to_double
        ) and not bodo.bodo_use_decimal:
            out_dtype = pa.float64()
        # Stick to existing decimal type in this case
        else:
            out_dtype = dtype

        new_col_info.append((idx, name, out_dtype))

    return new_col_info, None


def snowflake_type_str_to_pyarrow_datatype(
    types: set[str],
    cursor: SnowflakeCursor,
    sql_query: str,
    colname: str,
    source_colname: str,
    cur_type: str,
    tz: str,
    can_system_sample: bool,
) -> pa.DataType | None:
    """
    Convert a Set of Snowflake Type Strings to a PyArrow type

    Args:
        types: Set of Snowflake Type Strings to Convert to PyArrow
            Sometimes multiple types are coerced into one output
        cursor: Snowflake cursor to pass through for nested semi-structured types
        sql_query: Source table or query to get data for current column
            Used for nested semi-structured types
        colname: Current column name that we're typing
            Used for nested semi-structured types
        source_colname: Original column name from the source table / view
        cur_type: Currently known type of the column as a string
            Used for error reporting
        tz: System Timezone, for timestamp types
        can_system_sample: True if the table having its semi-structured types examined
            can be done with system sampling.

    Returns:
        PyArrow DataType if successful, None if unable to determine
    """

    # Always assume output is nullable for now
    types.discard("NULL_VALUE")

    # DOUBLE is floating-point (including NaN, Inf, ...)
    # DECIMAL is most non-integer numbers without explicit cast
    # INTEGER is whole numbers without explicit cast
    # For our case, we treat all decimal-point values as float64
    if "DOUBLE" in types:
        types.discard("DOUBLE")
        types.add("DECIMAL")
    # Snowflake auto-deletes zero value after decimal point, treating as integer
    # So we upcast to Float / Decimal, even across Integer -> Float
    if types == {"INTEGER", "DECIMAL"}:
        types = {"DECIMAL"}

    if len(types) > 1:
        return None
    if len(types) == 0:
        return pa.null()

    value_type = types.pop()
    if value_type == "DECIMAL":
        return pa.float64()
    elif value_type == "INTEGER":
        return pa.int64()
    elif value_type == "VARCHAR":
        return pa.large_string()
    elif value_type == "BINARY":
        return pa.binary()
    elif value_type in ("BOOL", "BOOLEAN"):
        return pa.bool_()
    elif value_type == "DATE":
        return pa.date32()
    elif value_type == "TIMESTAMP_NTZ":
        # TODO: Properly derive timestamp precision if necessary
        return pa.timestamp("ns")
    elif value_type in "TIMESTAMP_LTZ":
        # TODO: Properly derive timestamp precision if necessary
        return pa.timestamp("ns", tz=tz)
    elif value_type in "TIMESTAMP_TZ":
        return ArrowTimestampTZType()
    elif value_type == "ARRAY":
        return get_list_type_from_metadata(
            cursor,
            sql_query,
            colname,
            source_colname,
            cur_type,
            tz,
            can_system_sample,
        )
    elif value_type == "OBJECT":
        return get_map_type_from_metadata(
            cursor,
            sql_query,
            colname,
            source_colname,
            cur_type,
            tz,
            can_system_sample,
        )

    elif value_type == "VARIANT":
        raise BodoError(
            f"Bodo does not support reading VARIANT data found in column `{source_colname}`"
        )

    else:
        raise BodoError(f"Unknown Snowflake Type String: {value_type}")


def get_list_type_from_metadata(
    cursor: SnowflakeCursor,
    sql_query: str,
    cur_colname: str,
    source_colname: str,
    cur_type: str,
    tz: str,
    can_system_sample: bool,
):
    """
    Determine a precise output type for List Columns from Snowflake
    See snowflake_type_str_to_pyarrow_datatype for argument types
    """
    cur_type = cur_type.format("list[{}]")
    # TODO: Slice array for beginning content
    sample_clauses = None
    if can_system_sample:
        # If the source is a table, use system sampling to get 0.1% of all blocks which is a
        # quick way to reduce the total number of rows. Also filters to remove all rows where
        # the array is empty or null, so subsequent steps are only looking at rows that have
        # useful information.

        # However, we should only use this method of sampling if the row count is non-trivial
        # (e.g. above a thousand)
        sample_rate = get_sample_rate_for_system_sample(cursor, sql_query)
        if sample_rate is not None:
            sample_clauses = f"SAMPLE SYSTEM ({sample_rate}) WHERE {cur_colname} is not null and array_size({cur_colname}) > 0"

    if sample_clauses is None:
        # If the source is a view, do the same but using a large limit to quickly narrow down
        # the number of rows to sample from. System sampling is fast but doesn't work on views,
        # row sampling does work on views but is prohibitively slow, so limit is the only option.
        sample_clauses = f"WHERE {cur_colname} is not null and array_size({cur_colname}) > 0 LIMIT 1000000"

    # Use row sampling on the narrowed down result that does not contain empty/null arrays to
    # reduce the number of rows that are exploded to at most 1000
    narrowed_query = (
        f"(SELECT {cur_colname} FROM {sql_query} {sample_clauses}) SAMPLE (10000 ROWS)"
    )

    # Use flatten on the further reduced result to explode each array into 1 row per element,
    # then gather all of the distinct types of the inner elements.
    flatten_query = f"SELECT out.value as V FROM ({narrowed_query}), LATERAL FLATTEN(input => {cur_colname}) out"
    probe_query = f"SELECT DISTINCT TYPEOF(V) as VALUES_TYPE from ({flatten_query})"

    # Send the probe query until we breach the timeout, in case we are unlucky with sampling.
    start_time = time.time()
    while (time.time() - start_time) < SF_READ_SCHEMA_PROBE_TIMEOUT:
        probe_res = execute_query(
            cursor,
            probe_query,
            timeout=SF_READ_SCHEMA_PROBE_TIMEOUT,
        )
        if probe_res is not None and len(types_df := probe_res.fetch_pandas_all()) > 0:
            break
    # If the query failed or did not produce any type strings, indicate failure since
    # it will not be possible to infer the correct type.
    if probe_res is None or len(types_df) == 0:
        raise BodoError(
            f"Snowflake Probe Query Failed or Timed out While Typing List Content in Column {source_colname}. "
            f"It is currently statically typed as {cur_type.format('...')} in the Source:\n"
            f"{sql_query}"
        )

    value_types: set[str] = set(types_df["VALUES_TYPE"].to_list())
    pa_type = snowflake_type_str_to_pyarrow_datatype(
        value_types,
        cursor,
        f"({flatten_query})",
        "V",
        source_colname,
        cur_type,
        tz,
        False,  # The source is no longer a table since it is the output of the flatten query
    )

    if pa_type is None:
        raise BodoError(
            f"Snowflake Probe determined that Column {source_colname} in query or table:\n"
            f"{sql_query}\n"
            f"is type {cur_type.format('variant')}. We are unable to narrow the type further, because the `variant` "
            f"content has items of types {sorted(value_types)}. This indicated that the outer list is either:\n"
            "  - A variant / union array with multiple datatypes\n"
            "  - An array representing a tuple with a common schema across rows\n"
            "Bodo currently does not support either array types"
        )

    return pa.large_list(pa_type)


def get_sample_rate_for_system_sample(cursor: SnowflakeCursor, sql_query: str):
    """
    If the table produced by a sql query which is valid for system sampling is
    large enough that system sample should be done at all, returns the sample rate.
    Otherwise, returns None.
    """
    rowcount_res = execute_query(
        cursor,
        f"SELECT COUNT(*) FROM {sql_query}",
        timeout=SF_READ_SCHEMA_PROBE_TIMEOUT,
    )
    if rowcount_res is not None:
        rowcount_df = rowcount_res.fetch_pandas_all()
        row_count = rowcount_df["COUNT(*)"].iloc[0]
        if row_count < 10**3:
            return None
        elif row_count < 10**6:
            return 5
        elif row_count < 10**9:
            return 1
        elif row_count < 10**10:
            return 0.5
        elif row_count < 10**11:
            return 0.1
        else:
            return 0.05
    return None


def get_variant_type_from_metadata(
    cursor: SnowflakeCursor,
    sql_query: str,
    cur_colname: str,
    source_colname: str,
    tz: str,
    can_system_sample: bool,
):
    """
    Determine a precise output type for variant Columns from Snowflake
    See snowflake_type_str_to_pyarrow_datatype for argument types
    """
    sample_clauses = None
    if can_system_sample:
        # If the source is a table, use system sampling to get 0.1% of all blocks which is a
        # quick way to reduce the total number of rows. Also filters to remove all rows where
        # the value is null, so subsequent steps are only looking at rows that have
        # useful information.

        # However, we should only use this method of sampling if the row count is non-trivial
        # (e.g. above a thousand)
        sample_rate = get_sample_rate_for_system_sample(cursor, sql_query)
        if sample_rate is not None:
            sample_clauses = (
                f"SAMPLE SYSTEM ({sample_rate}) WHERE {cur_colname} is not null"
            )

    if sample_clauses is None:
        # If the source is a view, do the same but using a large limit to quickly narrow down
        # the number of rows to sample from. System sampling is fast but doesn't work on views,
        # row sampling does work on views but is prohibitively slow, so limit is the only option.
        sample_clauses = f"WHERE {cur_colname} is not null LIMIT 1000000"

    # Use row sampling on the narrowed down result that does not contain null rows to
    # reduce the number of rows that are exploded to at most 10000.
    narrowed_query = f"(SELECT {cur_colname} as V FROM {sql_query} {sample_clauses}) SAMPLE (10000 ROWS)"

    # Gather all of the distinct types of the reduced rows to infer the correct row type.
    # We need to wrap `V` in TO_VARIANT because `V` might not be a true variant
    # if it is actually a TimestampTZ that we casted to force a string
    # representation.
    probe_query = (
        f"SELECT DISTINCT TYPEOF(TO_VARIANT(V)) as VALUES_TYPE from ({narrowed_query})"
    )
    # Send the probe query until we breach the timeout (or at most 5 times),
    # in case we are unlucky with sampling.
    start_time = time.time()
    max_tries = 5
    tries = 0
    while (
        time.time() - start_time
    ) < SF_READ_SCHEMA_PROBE_TIMEOUT and tries < max_tries:
        tries += 1
        probe_res = execute_query(
            cursor,
            probe_query,
            timeout=SF_READ_SCHEMA_PROBE_TIMEOUT,
        )
        if probe_res is not None and len(types_df := probe_res.fetch_pandas_all()) > 0:
            break
    # If the query failed or did not produce any type strings, indicate failure since
    # it will not be possible to infer the correct type.
    if probe_res is None or len(types_df) == 0:
        # However, before failing, first check to see if there were any non-null rows.
        # If there were not, return a null datatype.
        null_query = f"SELECT COUNT({cur_colname}) as NON_NULL_COUNT FROM {sql_query}"
        count_res = execute_query(
            cursor,
            null_query,
            timeout=SF_READ_SCHEMA_PROBE_TIMEOUT,
        )
        if (
            count_res is not None
            and len(count_df := count_res.fetch_pandas_all()) > 0
            and count_df["NON_NULL_COUNT"].iloc[0] == 0
        ):
            if bodo.get_rank() == 0:
                warnings.warn(
                    BodoWarning(
                        f"The column {source_colname} is typed as a null array since the source is a variant column with no non-null entries."
                    )
                )
            return pa.null()
        else:
            raise BodoError(
                f"Snowflake Probe Query Failed or Timed out While Typing List Content in Column {source_colname}. "
                f"It is currently statically typed as VARIANT in the Source:\n"
                f"{sql_query}"
            )

    value_types: set[str] = set(types_df["VALUES_TYPE"].to_list())
    pa_type = snowflake_type_str_to_pyarrow_datatype(
        value_types,
        cursor,
        sql_query,
        cur_colname,
        source_colname,
        "variant",
        tz,
        can_system_sample,
    )

    if pa_type is None:
        raise BodoError(
            f"Snowflake Probe determined that Column {source_colname} in query or table:\n"
            f"{sql_query}\n"
            "is type variant. We are unable to narrow the type further, because the `variant` "
            f"content has items of types {sorted(value_types)}. This indicated that the column is "
            "a variant / union column with multiple datatypes.\n"
            "Bodo currently does not support this array type."
        )

    return pa_type


def get_struct_type_from_metadata(
    cursor: SnowflakeCursor,
    sql_query: str,
    cur_colname: str,
    source_colname: str,
    cur_type: str,
    tz: str,
):
    """
    For a potential Struct columns, determine a common set of field names
    and their internal types from Snowflake
    See snowflake_type_str_to_pyarrow_datatype for argument types
    """

    cur_type = cur_type.format("struct[{}]")
    # TODO: Improve potential performance of query
    probe_query = f"""\
        WITH source AS (
            SELECT
                {cur_colname} AS vals,
                COUNT(vals) OVER () as src_cnt
            FROM (SELECT * FROM ({sql_query}) WHERE {cur_colname} is not NULL LIMIT 1000)
        ),
        keys_table AS (
            SELECT distinct t.key::text as keys
            FROM source s, lateral flatten(input => vals) t
        )
        SELECT
            keys,
            COUNT(GET(vals, keys)),
            ANY_VALUE(src_cnt) as total,
            ARRAY_AGG(distinct TYPEOF(GET(vals, keys))) as types
        FROM keys_table, source
        GROUP BY keys
    """

    probe_res = execute_query(
        cursor,
        probe_query,
        timeout=SF_READ_SCHEMA_PROBE_TIMEOUT,
    )

    key_types: list[tuple[str, int, int, str]]
    if probe_res is None or len(key_types := probe_res.fetchall()) == 0:
        raise BodoError(
            f"Snowflake Probe Query Failed or Timed out While Typing Object Content in Column {source_colname}. "
            f"It is currently statically typed as {cur_type.format('...')}. Timed-out Query:\n"
            f"{sql_query}"
        )

    fields = []
    for key_name, cnt, total, types_list_str in key_types:
        # Edge case, only null columns have cnt == 0
        # Metric ignores null rows
        if cnt != 0 and cnt / total < 0.005:
            cur_type = cur_type.format("...")
            raise BodoError(
                f"Snowflake Probe found that Column {source_colname}, currently typed as {cur_type}, "
                f"has a field {key_name} in < 0.5% of non-null rows. This implies {source_colname} has object elements "
                "with heterogenous values, which Bodo does not currently support."
            )

        value_types: set[str] = set(json.loads(types_list_str))
        elem_query = f"SELECT GET({cur_colname}, '{key_name}') as V FROM ({sql_query})"
        pa_type = snowflake_type_str_to_pyarrow_datatype(
            value_types,
            cursor,
            f"({elem_query})",
            "V",
            source_colname,
            cur_type.format(f"... {key_name}: {{}} ..."),
            tz,
            False,  # The source is no longer a table since it is the output of the flatten query
        )
        if pa_type is None:
            raise BodoError(
                f"Snowflake Probe determined that Column {source_colname} in query or table:\n"
                f"{sql_query}\n"
                f"is type {cur_type.format(f'... {key_name}: variant ...')}. We are unable to narrow the type further, because "
                f"field {key_name} was found containing multiple types {sorted(value_types)}. "
                "Bodo currently does not support heterogenous column types."
            )

        fields.append(pa.field(key_name, pa_type, nullable=True))

    return pa.struct(fields)


def get_map_type_from_metadata(
    cursor: SnowflakeCursor,
    sql_query: str,
    cur_colname: str,
    source_colname: str,
    cur_type: str,
    tz: str,
    can_system_sample: bool,
):
    """
    Determine a precise output type for Map Columns from Snowflake
    See snowflake_type_str_to_pyarrow_datatype for argument types
    """
    sample_clauses = None
    if can_system_sample:
        # If the source is a table, use system sampling to get 0.1% of all blocks which is a
        # quick way to reduce the total number of rows. Also filters to remove all rows where
        # the array is empty or null, so subsequent steps are only looking at rows that have
        # useful information.

        # However, we should only use this method of sampling if the row count is non-trivial
        # (e.g. above a thousand)
        sample_rate = get_sample_rate_for_system_sample(cursor, sql_query)
        if sample_rate is not None:
            sample_clauses = f"SAMPLE SYSTEM ({sample_rate}) WHERE {cur_colname} is not null and array_size(object_keys({cur_colname})) > 0"

    if sample_clauses is None:
        # If the source is a view, do the same but using a large limit to quickly narrow down
        # the number of rows to sample from. System sampling is fast but doesn't work on views,
        # row sampling does work on views but is prohibitively slow, so limit is the only option.
        sample_clauses = f"WHERE {cur_colname} is not null and array_size(object_keys({cur_colname})) > 0 LIMIT 1000000"

    # Use row sampling on the narrowed down result that does not contain empty/null objects to
    # reduce the number of rows that are exploded to at most 1000
    narrowed_query = (
        f"(SELECT {cur_colname} FROM {sql_query} {sample_clauses}) SAMPLE (10000 ROWS)"
    )

    # Use flatten on the further reduced result to explode each array into 1 row per element,
    # then gather all of the distinct types of the inner elements.
    flatten_query = f"SELECT out.value as V  FROM ({narrowed_query}), LATERAL FLATTEN(input => {cur_colname}) out"
    probe_query = f"SELECT DISTINCT TYPEOF(V) as VALUES_TYPE from ({flatten_query})"

    # Send the probe query until we breach the timeout, in case we are unlucky with sampling.
    start_time = time.time()
    while (time.time() - start_time) < SF_READ_SCHEMA_PROBE_TIMEOUT:
        probe_res = execute_query(
            cursor,
            probe_query,
            timeout=SF_READ_SCHEMA_PROBE_TIMEOUT,
        )
        if probe_res is not None and len(types_df := probe_res.fetch_pandas_all()) > 0:
            break
    # If the query failed or did not produce any type strings, indicate failure since
    # it will not be possible to infer the correct type.
    if probe_res is None or len(types_df) == 0:
        raise BodoError(
            f"Snowflake Probe Query Failed or Timed out While Determining the Type of Column {cur_colname}.\n"
            f"Currently determined to be {cur_type.format('map[str, ...]')}. Timed-out Query:\n{sql_query}"
        )

    # If we can find a unified value type, we will read the object column as
    # maps with string keys and values of the unified value type.
    value_types: set[str] = set(types_df["VALUES_TYPE"].to_list())
    pa_type = snowflake_type_str_to_pyarrow_datatype(
        value_types,
        cursor,
        f"({flatten_query})",
        "V",
        source_colname,
        cur_type.format("map[str, {}]"),
        tz,
        False,  # The source is no longer a table since it is the output of the flatten query
    )

    # If we do not have a unified value type, try again but attempting
    # to infer the type as a struct column.
    if pa_type is None:
        return get_struct_type_from_metadata(
            cursor,
            sql_query,
            cur_colname,
            source_colname,
            cur_type,
            tz,
        )

    return pa.map_(pa.large_string(), pa_type)


def can_table_be_system_sampled(cursor: SnowflakeCursor, table_name: str | None):
    """
    Returns True if the table referenced by table_name is actually a table
    that can be sampled from using system sampling by sending off a system
    sample query that will not retrieve any real data.

    If nothing goes wrong, then the table is actually a table that can be
    system sampled, so we return True.

    If something goes wrong, the table cannot be system sampled so we
    return False.
    """
    if table_name is None:
        return False
    try:
        fake_sample_query = f"SELECT * FROM {table_name} SAMPLE SYSTEM(0)"
        result = execute_query(
            cursor, fake_sample_query, timeout=SF_READ_SCHEMA_PROBE_TIMEOUT
        )
        if result is None:
            return False
        result.fetch_pandas_all()
        return True
    except snowflake.connector.errors.ProgrammingError:
        return False


def get_schema_from_metadata(
    cursor: SnowflakeCursor,
    sql_query: str,
    orig_table: str | None,
    orig_table_indices: tuple[int, ...] | None,
    is_select_query: bool,
    is_table_input: bool,
    downcast_decimal_to_double: bool,
) -> tuple[
    list[pa.Field], list, list[bool], list[int], list[pa.DataType], list[str] | None
]:  # pragma: no cover
    """
    Determine the Arrow schema and Bodo types of the query output
    The approach is described in a Confluence Doc:
    https://bodo.atlassian.net/wiki/spaces/B/pages/1238433836/Snowflake+Read+Table+Schema+Inference
    This function executes independently on ranks.

    Args:
        cursor: Snowflake Cursor to Perform Operations in
        sql_query: Base SQL Query Operation
        orig_table: Passed to (and see) get_number_types_from_metadata
        orig_table_indices: Passed to (and see) get_number_types_from_metadata
        is_select_query: sql_query is a SELECT query
        downcast_decimal_to_double: Passed to (and see) get_number_types_from_metadata

    Returns:
        pa_fields: List of PyArrow Fields for Each Column
            Contains source column name, type, and nullability
        col_types: List of Output Bodo Types for Each Column
        check_dict_encoding: Should we check dictionary encoding for this column?
        unsupported_columns: Output Column Names with Unsupported Types
            Bodo can't read the column in but should still recognize it for
            other uses, like column pruning
        unsupported_arrow_types: Arrow Types of Each Unsupported Column
    """
    # Get Snowflake Metadata for Query
    # Use it to determine the general / broad Snowflake types
    # The actual Arrow result may use smaller types for columns (initially int64, use int8)
    desc_query = f"select * from {sql_query}" if is_table_input else sql_query
    query_field_metadata = cursor.describe(desc_query)
    # Session Timezone, should be populated by the describe operation
    tz: str = cursor._timezone  # type: ignore

    # Equivalent PyArrow Fields
    arrow_dtypes: list[tuple[str, pa.DataType, bool]] = []

    for i, field_meta in enumerate(query_field_metadata):
        dtype = type_code_to_arrow_type(
            field_meta.type_code, field_meta, tz, is_select_query
        )
        is_nullable = field_meta.is_nullable
        # For any UnknownSnowflakeType columns, fetch metadata to get internal
        if isinstance(dtype, UnknownSnowflakeType):
            can_system_sample = can_table_be_system_sampled(cursor, orig_table)
            if orig_table is None:
                src, cur_col = f"({desc_query})", f"${i + 1}"
            else:
                src, cur_col = orig_table, f"${orig_table_indices[i] + 1}"
            if dtype == UnknownSnowflakeType.LIST:
                dtype = get_list_type_from_metadata(
                    cursor,
                    src,
                    cur_col,
                    field_meta.name,
                    "{}",
                    tz,
                    can_system_sample,
                )
            elif dtype == UnknownSnowflakeType.OBJECT:
                dtype = get_map_type_from_metadata(
                    cursor,
                    src,
                    cur_col,
                    field_meta.name,
                    "{}",
                    tz,
                    can_system_sample,
                )
            elif dtype == UnknownSnowflakeType.VARIANT:
                dtype = get_variant_type_from_metadata(
                    cursor,
                    src,
                    cur_col,
                    field_meta.name,
                    tz,
                    can_system_sample,
                )

        # We may set dtype to pa.null() in get_variant_type_from_metadata if the
        # data is empty, but the actual column's metadata may be set to
        # non-nullable.
        # Setting the nullable flag to True is necessary to avoid Arrow errors.
        # See https://bodo.atlassian.net/browse/BSE-2918?focusedCommentId=29750
        if dtype == pa.null():
            is_nullable = True

        assert isinstance(dtype, pa.DataType), (
            "All Snowflake Columns Should Have a PyArrow DataType by Now"
        )
        arrow_dtypes.append((field_meta.name, dtype, is_nullable))

    # For any NUMBER columns, fetch SYSTEM$TYPEOF metadata to determine
    # the smallest viable integer type (number of bytes)
    schema_timeout_info = None
    number_cols = [
        (i, name, d)
        for i, (name, d, _) in enumerate(arrow_dtypes)
        if isinstance(d, pa.DataType) and pa.types.is_decimal(d)
    ]
    if is_select_query and len(number_cols) != 0:
        out_number_cols, schema_timeout_info = get_number_types_from_metadata(
            cursor,
            sql_query,
            orig_table,
            orig_table_indices,
            downcast_decimal_to_double,
            number_cols,
        )
        for i, name, d in out_number_cols:
            arrow_dtypes[i] = (
                name,
                _try_convert_decimal_type_to_integer_type(d),
                arrow_dtypes[i][2],
            )

    # By this point, we should have fixed data types for all columns
    arrow_fields: list[pa.Field] = []
    for name, d, nullable in arrow_dtypes:
        arrow_fields.append(pa.field(name, d, nullable))

    # Convert Arrow Types to Bodo Types
    col_types = []
    unsupported_columns = []
    unsupported_arrow_types = []
    for i, field in enumerate(arrow_fields):
        dtype, supported = _get_numba_typ_from_pa_typ(
            field,
            False,  # index_col
            field.nullable,  # nullable_from_metadata
            None,  # category_info
        )
        col_types.append(dtype)
        if not supported:
            unsupported_columns.append(i)
            # Store the unsupported arrow type for future error messages
            unsupported_arrow_types.append(field.type)

    return (
        arrow_fields,
        col_types,
        [pa.types.is_string(f.type) for f in arrow_fields],
        unsupported_columns,
        unsupported_arrow_types,
        schema_timeout_info,
    )


def _get_table_row_count(cursor: SnowflakeCursor, table_name: str) -> int | None:
    """get total number of rows for a Snowflake table. Returns None if input is not a
    table or probe query failed.

    Args:
        cursor: Snowflake connector connection cursor object
        table_name: table name

    Returns:
        number of rows or None if failed
    """
    count_res = execute_query(
        cursor,
        f"select count(*) from ({table_name})",
        timeout=SF_READ_DICT_ENCODING_PROBE_TIMEOUT,
    )
    if count_res is None:
        return None

    total_rows = count_res.fetchall()[0][0]
    return total_rows


def _detect_column_dict_encoding(
    sql_query,
    query_args,
    string_col_ind,
    undetermined_str_cols,
    cursor,
    col_types,
    is_table_input,
    schema_name: str | None,
):
    """Detects Snowflake columns that need to be dictionary-encoded using a query that
    gets approximate data cardinalities.

    Args:
        sql_query (str): read query or Snowflake table name
        query_args (list(str)): probe query arguments, e.g. ['approx_count_distinct($1)',
            'approx_count_distinct($2)']
        string_col_ind (list(int)): index of string columns in col_types
        undetermined_str_cols (iterable(str)): column names of string columns that need
            dict-encoding probe (not manually specified)
        cursor (SnowflakeCursor): Snowflake connector connection cursor object
        col_types (list(types.Type)): read data types to update with dict-encoding info
        is_table_input (bool): read query is a table name
        schema_name (Optional[str]): schema name (if table input). This is passed
            separately so we can find the table if its not in the default schema.

    Returns:
        Optional[tuple[int, list[str]]]: debug info if the probe query timed out
    """
    table_name_query = sql_query
    # Recreate the full path to the table if we have passed in the schema
    if schema_name is not None and is_table_input:
        sql_query = f"{schema_name}.{sql_query}"

    # Determine if the string columns are dictionary encoded
    dict_encode_timeout_info: tuple[int, list[str]] | None = None

    # the limit on the number of rows total to read for the probe
    probe_limit = max(SF_READ_DICT_ENCODING_PROBE_ROW_LIMIT // len(query_args), 1)

    # Always read the total rows in case we have a view. In the query is complex
    # this may time out.
    total_rows = _get_table_row_count(cursor, sql_query)

    if total_rows is not None and total_rows <= SF_SMALL_TABLE_THRESHOLD:
        # use dict-encoding for all strings if we have a small table since it can be
        # joined with large tables producing many duplicates (dict-encoding overheads
        # are minimal for small tables if this is not true).
        for i in string_col_ind:
            col_types[i] = dict_str_arr_type
        return dict_encode_timeout_info

    # make sure table_name is an actual table and not a view since system sampling
    # doesn't work on views
    is_view = not is_table_input
    if is_table_input and total_rows is not None:
        if schema_name is not None:
            schema_info = f"in SCHEMA {schema_name}"
        else:
            schema_info = ""
        # Remove any quotes for case sensitivity since the table
        # name is passed as a string instead. The quotes will give
        # false negatives.
        table_name = table_name_query.replace('"', "")
        check_res = execute_query(
            cursor,
            # Note we need both like and starts with because in this context
            # like is case-insensitive but starts with is case-sensitive. Since they
            # are exactly the same this will only match the exact query.
            # See https://bodo.atlassian.net/browse/BSE-277 for why this is necessary.
            f"show tables like '{table_name}' {schema_info} starts with '{table_name}'",
            timeout=SF_READ_DICT_ENCODING_PROBE_TIMEOUT,
        )
        if check_res is None or not check_res.fetchall():
            # empty output means view
            is_view = True

    # use system sampling if input is a table
    if is_table_input and not is_view and total_rows is not None:
        # get counts for roughly probe_limit rows to minimize overheads
        sample_percentage = (
            0 if total_rows <= probe_limit else probe_limit / total_rows * 100
        )
        sample_call = (
            f"SAMPLE SYSTEM ({sample_percentage})" if sample_percentage else ""
        )
        predict_cardinality_call = (
            f"select count(*),{', '.join(query_args)} from {sql_query} {sample_call}"
        )
        if bodo.user_logging.get_verbose_level() >= 2:
            encoding_msg = "Using Snowflake system sampling for dictionary-encoding detection:\nQuery: %s\n"
            bodo.user_logging.log_message(
                "Dictionary Encoding",
                encoding_msg,
                predict_cardinality_call,
            )
    else:
        # get counts for roughly probe_limit rows to minimize overheads
        if total_rows is not None:
            sample_percentage = (
                0 if total_rows <= probe_limit else probe_limit / total_rows * 100
            )
            sample_call = f"SAMPLE ({sample_percentage})" if sample_percentage else ""
        else:
            sample_call = f"limit {probe_limit}"

        # construct the prediction query script for the string columns
        # in which we sample 1 percent of the data
        # upper bound limits the total amount of sampling that will occur
        # to prevent a hang/timeout
        predict_cardinality_call = (
            f"select count(*),{', '.join(query_args)}"
            f"from ( select * from ({sql_query}) {sample_call})"
        )

    prediction_query = execute_query(
        cursor,
        predict_cardinality_call,
        timeout=SF_READ_DICT_ENCODING_PROBE_TIMEOUT,
    )

    if prediction_query is None:  # pragma: no cover
        # It is hard to get Snowflake to consistently
        # and deterministically time out, so this branch
        # isn't tested in the unit tests.
        dict_encode_timeout_info = (probe_limit, list(undetermined_str_cols))
        if SF_READ_DICT_ENCODING_IF_TIMEOUT:
            for i in string_col_ind:
                col_types[i] = dict_str_arr_type

    else:
        cardinality_data: pa.Table = prediction_query.fetch_arrow_all()  # type: ignore
        # calculate the level of uniqueness for each string column
        total_rows = cardinality_data[0][0].as_py()
        n_uniques = [
            cardinality_data[i][0].as_py() for i in range(1, len(query_args) + 1)
        ]
        # filter the string col indices based on the criterion
        n_rows = max(total_rows, 1)
        col_inds_to_convert = filter(
            lambda x: (x[0] / n_rows) <= SF_READ_DICT_ENCODE_CRITERION
            and (
                (not bodo.bodosql_use_streaming_plan)
                or x[0] < SF_STREAM_READ_DICT_ENCODE_LIMIT
            ),
            zip(n_uniques, string_col_ind),
        )
        for _, ind in col_inds_to_convert:
            col_types[ind] = dict_str_arr_type

    return dict_encode_timeout_info


def get_schema(
    conn: SnowflakeConnection,
    sql_query: str,
    is_select_query: bool,
    is_table_input: bool,
    _bodo_read_as_dict: list[str] | None,
    downcast_decimal_to_double: bool,
    orig_table: str | None = None,
    orig_table_indices: tuple[int, ...] | None = None,
    convert_snowflake_column_names: bool = True,
):  # pragma: no cover
    """
    Args:
        conn (SnowflakeConnection): The connection being used to connect to the database
        sql_query (str): read query or Snowflake table name
        is_select_query (bool): TODO: document this
        is_table_input (bool): read query is a table name
        _bodo_read_as_dict (bool): Read all string columns as dict encoded strings.
        downcast_decimal_to_double: Passed to (and see) get_schema_from_metadata
        orig_table: Passed to (and see) get_schema_from_metadata
        orig_table_indices: Passed to (and see) get_schema_from_metadata
        convert_snowflake_column_names (bool, default True): Should Snowflake column names be
            converted to match SqlAlchemy. This is needed to ensure table path is consistent for
            casing with the SnowflakeCatalog.

    Returns:
        A large tuple containing: (#TODO: document this)
    """
    cursor = conn.cursor()

    (
        pa_fields,
        col_types,
        check_dict_encoding,
        unsupported_columns,
        unsupported_arrow_types,
        schema_timeout_info,
    ) = get_schema_from_metadata(
        cursor,
        sql_query,
        orig_table,
        orig_table_indices,
        is_select_query,
        is_table_input,
        downcast_decimal_to_double,
    )

    str_as_dict_cols = _bodo_read_as_dict if _bodo_read_as_dict else []
    str_col_name_to_ind = {}
    for i, check_dict_encoding in enumerate(check_dict_encoding):
        if check_dict_encoding:
            str_col_name_to_ind[pa_fields[i].name] = i

    # Map the snowflake original column name to the name that
    # is used from Python. This is used for comparing with
    # _bodo_read_as_dict which will use Python's convention.
    snowflake_case_map = {
        (
            name.lower() if convert_snowflake_column_names and name.isupper() else name
        ): name
        for name in str_col_name_to_ind.keys()
    }

    # If user-provided list has any columns that are not string
    # type, show a warning.
    non_str_columns_in_read_as_dict_cols = str_as_dict_cols - snowflake_case_map.keys()
    if len(non_str_columns_in_read_as_dict_cols) > 0:
        if bodo.get_rank() == 0:
            warnings.warn(
                BodoWarning(
                    f"The following columns are not of datatype string and hence cannot be read with dictionary encoding: {non_str_columns_in_read_as_dict_cols}"
                )
            )
    convert_dict_col_names = snowflake_case_map.keys() & str_as_dict_cols
    for name in convert_dict_col_names:
        col_types[str_col_name_to_ind[snowflake_case_map[name]]] = dict_str_arr_type

    query_args, string_col_ind = [], []
    undetermined_str_cols = snowflake_case_map.keys() - str_as_dict_cols
    for name in undetermined_str_cols:
        idx = str_col_name_to_ind[snowflake_case_map[name]]
        if orig_table_indices is None:
            col_name = f'"{snowflake_case_map[name]}"'
        else:
            # Use $ syntax to avoid directly using names.
            # Note: Snowflake/SQL is 1 indexed
            col_name = f"${orig_table_indices[idx] + 1}"

        query_args.append(f"approx_count_distinct({col_name})")
        string_col_ind.append(str_col_name_to_ind[snowflake_case_map[name]])

    # Determine if the string columns are dictionary encoded
    dict_encode_timeout_info: tuple[int, list[str]] | None = None

    if len(query_args) != 0 and SF_READ_AUTO_DICT_ENCODE_ENABLED:
        if orig_table != None:
            # The original table should be passed as database_name.schema.table_name.
            # Therefore, we split the . here.
            database_name, schema_name, table_name = orig_table.split(".")
            dict_encode_timeout_info = _detect_column_dict_encoding(
                table_name,
                query_args,
                string_col_ind,
                undetermined_str_cols,
                cursor,
                col_types,
                True,  # is_table_input
                # We pass the schema with the full path.
                f"{database_name}.{schema_name}",
            )
        else:
            dict_encode_timeout_info = _detect_column_dict_encoding(
                sql_query,
                query_args,
                string_col_ind,
                undetermined_str_cols,
                cursor,
                col_types,
                is_table_input,
                None,
            )

    # Ensure column name case matches Pandas/sqlalchemy. See:
    # https://github.com/snowflakedb/snowflake-sqlalchemy#object-name-case-handling
    # If a name is returned as all uppercase by the Snowflake connector
    # it means it is case insensitive or it was inserted as all
    # uppercase with double quotes. In both of these situations
    # pd.read_sql() returns the name with all lower case
    final_colnames: list[str] = []
    converted_colnames = set()
    for x in pa_fields:
        if convert_snowflake_column_names and x.name.isupper():
            converted_colnames.add(x.name.lower())
            final_colnames.append(x.name.lower())
        else:
            final_colnames.append(x.name)
    df_type = DataFrameType(data=tuple(col_types), columns=tuple(final_colnames))

    return (
        df_type,
        converted_colnames,
        unsupported_columns,
        unsupported_arrow_types,
        pa.schema(pa_fields),
        schema_timeout_info,
        dict_encode_timeout_info,
    )


class SnowflakeDataset:
    """Store dataset info in the way expected by Arrow reader in C++."""

    def __init__(self, batches: list[ResultBatch], schema, conn: SnowflakeConnection):
        # pieces, _bodo_total_rows and _bodo_total_rows are the attributes
        # expected by ArrowDataFrameReader, schema is for SnowflakeReader.
        # NOTE: getting this information from the batches is very cheap and
        # doesn't involve pulling data from Snowflake
        self.pieces = batches
        self._bodo_total_rows = 0
        for b in batches:
            b._bodo_num_rows = b.rowcount  # type: ignore
            self._bodo_total_rows += b._bodo_num_rows  # type: ignore
        self.schema = schema
        self.conn = conn
        # We have exact row counts (after filtering).
        self.row_level = True


class FakeArrowJSONResultBatch:
    """
    Results Batch used to return a JSONResult in arrow format while
    conforming to the same APIS as ArrowResultBatch
    """

    def __init__(self, json_batch: JSONResultBatch, schema: pa.Schema) -> None:
        self._json_batch = json_batch
        self._schema = schema

    @property
    def rowcount(self):
        return self._json_batch.rowcount

    def to_arrow(self, _: SnowflakeConnection | None = None) -> pa.Table:
        """
        Return the data in arrow format.

        Args:
            conn: Connection that is accepted by ArrowResultBatch. We ignore
                this argument but conform to the same API.

        Returns:
            The data in arrow format
        """
        # Iterate over the data to use the pa.Table.from_pylist
        # constructor
        pylist = []
        for row in self._json_batch.create_iter():
            # TODO: Check if isinstance(row, Exception) and handle somehow
            pylist.append(
                {self._schema.names[i]: col_val for i, col_val in enumerate(row)}  # type: ignore
            )
        table = pa.Table.from_pylist(pylist, schema=self._schema)
        return table


def set_timestamptz_format_connection_parameter_if_required(
    conn: SnowflakeConnection, schema: pa.Schema
):
    """
    Sets the connection parameter for timestamptz formatting if required.

    Args:
        conn: Snowflake connection
        schema: Arrow schema
    """

    param_required = False
    for type_ in schema.types:
        if isinstance(type_, bodo.hiframes.timestamptz_ext.ArrowTimestampTZType):
            param_required = True
            break

    desired_format = "YYYY-MM-DD HH24:MI:SS.FF9 TZH:TZM"

    # Read the TIMESTAMP_TZ_OUTPUT_FORMAT parameter if present
    original_param = (
        conn.cursor()
        .execute("show parameters like 'TIMESTAMP_TZ_OUTPUT_FORMAT'")
        .fetchone()[1]
    )
    if original_param and original_param != desired_format:
        warnings.warn(
            BodoWarning(
                f"TIMESTAMP_TZ_OUTPUT_FORMAT is set to {original_param}. "
                f"Overriding it to {desired_format}."
            )
        )

    if param_required:
        conn.cursor().execute(
            f"alter session set TIMESTAMP_TZ_OUTPUT_FORMAT = '{desired_format}'"
        )


def execute_length_query_helper(
    conn: SnowflakeConnection, query: str
) -> tuple[int, int]:
    """
    Helper function for the 'get_dataset' function. This
    executes the given query that is expected to return the
    number of rows in a table or output of a query.
    NOTE: The function must acts "independently", i.e. it will
    be executed on all ranks that it's called on and will do
    no error synchronization. The caller must handle those.

    Args:
        conn (SnowflakeConnection): Snowflake connection to use for
            executing the query.
        query (str): The query to execute.

    Returns:
        int: The number of rows returned by Snowflake.
        int: Time (in microseconds) to execute the query in Snowflake.
    """
    cur = conn.cursor()
    t0 = time.monotonic()
    cur.execute(query)
    sf_exec_time = time.monotonic() - t0
    if bodo.user_logging.get_verbose_level() >= 2:
        bodo.user_logging.log_message(
            "Snowflake Query Submission (Read)",
            "/* execute_length_query */ Snowflake Query ID: "
            + cur.sfqid
            + "\nSQL Text:\n"
            + query
            + f"\nApproximate Execution Time: {sf_exec_time:.3f}s",
        )
    # We are just loading a single row of data so we can just load
    # all of the data.
    arrow_data = cur.fetch_arrow_all()
    num_rows = arrow_data[0][0].as_py()  # type: ignore
    assert isinstance(num_rows, int), (
        f"Expected 'num_rows' to be an int, but got {type(num_rows)} instead."
    )
    cur.close()
    return num_rows, int(sf_exec_time * 1e6)


def execute_query_helper(
    conn: SnowflakeConnection,
    query: str,
    is_select_query: bool,
    schema: pa.Schema,
) -> tuple[int, list[ArrowResultBatch | FakeArrowJSONResultBatch], int]:
    """
    Helper function for 'get_dataset' to execute a query in Snowflake
    and return the Arrow batches of the result set and number of rows
    that are in the result set.
    NOTE: The function must acts "independently", i.e. it will
    be executed on all ranks that it's called on and will do
    no error synchronization. The caller must handle those.

    Returns:
        num_rows (int), batches (list), sf_exec_time (int):
            - Number of rows in the result set.
            - The result set in the form of list of either ArrowResultBatches
              or FakeArrowJSONResultBatch-es.
            - Time (in microseconds) to execute the query in Snowflake.
    """
    from snowflake.connector.result_batch import ArrowResultBatch, JSONResultBatch

    cur = conn.cursor()
    t0 = time.monotonic()
    cur.execute(query)
    sf_exec_time = time.monotonic() - t0
    # Fetch the total number of rows that will be loaded globally
    num_rows: int = cur.rowcount  # type: ignore
    assert isinstance(num_rows, int), (
        f"Expected 'num_rows' to be an int, but got {type(num_rows)} instead."
    )
    if bodo.user_logging.get_verbose_level() >= 2:
        bodo.user_logging.log_message(
            "Snowflake Query Submission (Read)",
            "/* execute_query */ Snowflake Query ID: "
            + cur.sfqid
            + "\nSQL Text:\n"
            + query
            + f"\nApproximate Execution Time: {sf_exec_time:.3f}s"
            + f"\nNumber of rows produced: {num_rows:,}",
        )

    # Get the list of result batches (this doesn't load data).
    batches: list[ResultBatch] = cur.get_result_batches()  # type: ignore
    assert isinstance(batches, list), (
        f"Expected 'batches' to be a list, but got {type(batches)} instead."
    )

    if len(batches) > 0 and not isinstance(batches[0], ArrowResultBatch):
        if (
            not is_select_query
            and len(batches) == 1
            and isinstance(batches[0], JSONResultBatch)
        ):
            # When executing a non-select query (e.g. DELETE), we may not obtain
            # the result in Arrow format and instead get a JSONResultBatch. If so
            # we convert the JSONResultBatch to a fake arrow that supports the same
            # APIs.
            #
            # To be conservative against possible performance issues during development, we
            # only allow a single batch. Every query that is currently supported only returns
            # a single row.
            batches = [FakeArrowJSONResultBatch(x, schema) for x in batches]  # type: ignore
        else:
            raise BodoError(
                f"Batches returns from Snowflake don't match the expected format. Expected Arrow batches but got {type(batches[0])}"
            )
    elif not all(isinstance(batch, ArrowResultBatch) for batch in batches):
        batches_types = [type(batch) for batch in batches]
        raise BodoError(
            f"Not all batch objects are ArrowResultBatches! batches types: {batches_types}"
        )

    cur.close()
    return num_rows, batches, int(sf_exec_time * 1e6)


def get_dataset(
    query: str,
    conn_str: str,
    schema: pa.Schema,
    only_fetch_length: bool = False,
    is_select_query: bool = True,
    is_parallel: bool = True,
    is_independent: bool = False,
) -> tuple[SnowflakeDataset, int]:  # pragma: no cover
    """Get snowflake dataset info required by Arrow reader in C++ and execute
    the Snowflake query

    Args:
        query: Query to execute inside Snowflake
        conn_str: Connection string Bodo will parse to connect to Snowflake.
        only_fetch_length (bool, optional): Is the query just used to fetch rather
            than return a table? If so we just run a COUNT(*) query and broadcast
            the length without any batches.. Defaults to False.
        is_select_query (bool, optional): Is this query a select?
        is_parallel (bool, optional): Is the output data distributed?
        is_independent(bool, optional): Is this called by all ranks independently
        (e.g. distributed=False)?

    Raises:
        BodoError: Raises an error if Bodo returns the data in the wrong format.

    Returns:
        Returns a pair of values:
            - The SnowflakeDataset object that holds the information to access
              the actual data results.
            - The number of rows in the output.
    """
    assert not (only_fetch_length and not is_select_query), (
        "The only length optimization can only be run with select queries"
    )

    # Data cannot be distributed if each rank is independent
    assert not (is_parallel and is_independent), (
        "Snowflake get_dataset: is_parallel and is_independent cannot be True at the same time"
    )

    # Snowflake import
    try:
        import snowflake.connector  # noqa
    except ImportError:
        raise BodoError(
            "Snowflake Python connector packages not found. "
            "Fetching data from Snowflake requires snowflake-connector-python. "
            "This can be installed by calling 'conda install -c conda-forge snowflake-connector-python' "
            "or 'pip install snowflake-connector-python'."
        )

    ev = tracing.Event("get_snowflake_dataset", is_parallel=is_parallel)

    comm = MPI.COMM_WORLD

    # Connect to Snowflake. This is the same connection that will be used
    # to read data.
    # We only trace on rank 0 (is_parallel=False) because we want 0 to start
    # executing the queries as soon as possible (don't sync event)
    err_connecting: Exception | None = None
    try:
        conn = snowflake_connect(conn_str, is_parallel=False)
        # Set the TIMESTAMP_TZ_OUTPUT_FORMAT parameter if required
        set_timestamptz_format_connection_parameter_if_required(conn, schema)
    except Exception as e:
        err_connecting = e

    # Check if this failed on any rank.
    bodo.spawn.utils.sync_and_reraise_error(
        err_connecting,
        _is_parallel=(not is_independent),
        # We don't broadcast the errors in case they are not pickle-able.
        bcast_lowest_err=False,
        default_generic_err_msg=(
            "Snowflake get_dataset: One or more ranks failed to connect to "
            "Snowflake. See error(s) on the other ranks."
        ),
    )

    # Number of rows loaded. This is only used if we are loading
    # 0 columns
    num_rows: int = -1
    sf_exec_time: int = 0
    batches: list[ArrowResultBatch | FakeArrowJSONResultBatch] = []
    error: Exception | None = None

    # The control flow for the below if-conditional clause:
    #   1. If the rank is 0 or the ranks are independent from each other, i.e. each rank is executing the function independently, execute the query
    #   2. If the ranks are not independent from each other, we want to broadcast the output to all ranks
    if only_fetch_length and is_select_query:
        # If we are loading 0 columns, the query will just be a COUNT(*).
        # In this case we can skip computing the query
        # not is_parallel is needed here to handle cases where read is called by one rank
        # with (distributed=False)
        # NOTE: it'll be unnecessary in case replicated case
        # and read is called by all ranks but we opted for that to simplify compiler work.
        if bodo.get_rank() == 0 or is_independent:
            try:
                num_rows, sf_exec_time = execute_length_query_helper(conn, query)
            except Exception as e:
                error = e
    else:
        # We need to actually submit a Snowflake query
        if bodo.get_rank() == 0 or is_independent:
            try:
                # Execute query
                num_rows, batches, sf_exec_time = execute_query_helper(
                    conn, query, is_select_query, schema
                )
            except Exception as e:
                error = e

    bodo.spawn.utils.sync_and_reraise_error(
        error,
        (not is_independent),
        # In case the error is not pickle-able, we only raise it on rank 0.
        bcast_lowest_err=False,
        # On all other ranks, we will raise a generic exception.
        default_generic_err_msg="Exception encountered while reading from Snowflake. See rank 0 for more details.",
    )

    # If the ranks are not independent from each other, broadcast num_rows
    if not is_independent:
        num_rows = comm.bcast(num_rows)
        batches = comm.bcast(batches)  # NOP in the 'only_fetch_length' case.

    # Fix for a Memory Leak in Streaming with CREATE TABLE LIKE or LIMIT 0
    # TODO: Using HPy in C++ should make this unnecessary
    if num_rows == 0:
        batches = []

    ds = SnowflakeDataset(batches, schema, conn)
    ev.finalize()
    return ds, num_rows, sf_exec_time


# --------------------------- snowflake_write helper functions ----------------------------
def create_internal_stage(
    cursor: SnowflakeCursor, is_temporary: bool = False
) -> str:  # pragma: no cover
    """Create an internal stage within Snowflake. If `is_temporary=False`,
    the named stage must be dropped manually in `drop_internal_stage()`

    Args
        cursor: Snowflake connection cursor
        is_temporary: Whether the created stage is temporary.
            Named stages are suitable for data loads that could involve multiple users:
            https://docs.snowflake.com/en/user-guide/data-load-local-file-system-create-stage.html#named-stages.
            From experimentation, temporary stages are only accessible to the cursor
            that created them, and are not suitable for this operation which involves
            multiple simultaneous uploads from different connections.

    Returns
        stage_name: Name of created internal stage
    """
    ev = tracing.Event("create_internal_stage", is_parallel=False)

    # Snowflake import
    try:
        import snowflake.connector
    except ImportError:
        raise BodoError(
            "Snowflake Python connector packages not found. "
            "Using 'to_sql' with Snowflake requires snowflake-connector-python. "
            "This can be installed by calling 'conda install -c conda-forge snowflake-connector-python' "
            "or 'pip install snowflake-connector-python'."
        )

    stage_name = ""  # forward declaration
    stage_name_err = None  # forward declaration

    # We will quickly generate a stage name that doesn't already exist within Snowflake.
    # An infinite loop here is extremely unlikely unless uuid4's are used up.
    while True:
        try:
            stage_name = f"bodo_io_snowflake_{uuid4()}"
            if is_temporary:
                create_stage_cmd = "CREATE TEMPORARY STAGE"
            else:
                create_stage_cmd = "CREATE STAGE"

            create_stage_sql = (
                f'{create_stage_cmd} "{stage_name}" '
                f"/* io.snowflake.create_internal_stage() */ "
            )
            cursor.execute(create_stage_sql, _is_internal=True).fetchall()  # type: ignore
            break

        except snowflake.connector.ProgrammingError as pe:
            if pe.msg is not None and pe.msg.endswith("already exists."):
                continue
            stage_name_err = pe.msg
            break

    ev.finalize()

    if stage_name_err is not None:
        raise snowflake.connector.ProgrammingError(stage_name_err)
    return stage_name


def drop_internal_stage(cursor: SnowflakeCursor, stage_name: str):  # pragma: no cover
    """Drop an internal stage within Snowflake.

    Args
        cursor: Snowflake connection cursor
        stage_name: Name of internal stage to drop
    """
    ev = tracing.Event("drop_internal_stage", is_parallel=False)

    drop_stage_sql = (
        f'DROP STAGE IF EXISTS "{stage_name}" /* io.snowflake.drop_internal_stage() */ '
    )
    cursor.execute(drop_stage_sql, _is_internal=True)

    ev.finalize()


def do_upload_and_cleanup(
    cursor: SnowflakeCursor,
    chunk_idx: int,
    chunk_path: str,
    stage_name: str,
    stage_dir=None,
):  # pragma: no cover
    """Upload the parquet file at the given file stream or path to Snowflake
    internal stage in a parallel thread, and perform needed cleanup.
    Args
        cursor: Snowflake connection cursor
        chunk_idx: Index of the current parquet chunk
        chunk_path: Path to the file to upload
        stage_name: Snowflake internal stage name to upload files to
        stage_dir (str or None): Optionally, specify a directory within internal stage
    Returns None.
    """

    ev_upload_parquet = tracing.Event(
        f"upload_parquet_file{chunk_idx}", is_parallel=False
    )

    if stage_dir is None:
        stage_name_with_dir = f'@"{stage_name}"'
    else:
        stage_name_with_dir = f'@"{stage_name}"/{stage_dir}/'

    # Windows "\" should be replaced with "/" for Snowflake PUT command when using quotes:
    # https://docs.snowflake.com/en/sql-reference/sql/put
    chunk_path = chunk_path.replace("\\", "/")

    upload_sql = (
        f"PUT 'file://{chunk_path}' {stage_name_with_dir} AUTO_COMPRESS=FALSE "
        f"/* io.snowflake.do_upload_and_cleanup() */"
    )
    cursor.execute(upload_sql, _is_internal=True).fetchall()  # type: ignore
    ev_upload_parquet.finalize()

    # Remove chunk file
    os.remove(chunk_path)


def create_table_handle_exists(
    cursor: SnowflakeCursor,
    location: str,
    sf_schema,
    if_exists: str,
    table_type: str,
    always_escape_col_names=False,
    create_table_info=None,
):  # pragma: no cover
    """Automatically create a new table in Snowflake at the given location if
    it doesn't exist, following the schema of staged files.
    Note: This is intended to be called only from Rank 0.

    Args
        cursor: Snowflake connection cursor
        location: Location to create a table
        sf_schema (dict): key: dataframe column names, value: dataframe column snowflake datatypes
        if_exists: Action to take if table already exists:
            "fail": If table exists, raise a ValueError. Create if does not exist
            "replace": If table exists, drop it, recreate it, and insert data.
                Create if does not exist
            "append": If table exists, insert data. Create if does not exist
        table_type: Type of table to create. Must be one of "", "TRANSIENT", or "TEMPORARY"
        always_escape_col_names: True if we are in BodoSQL table write, which allows always escaping
            column names since BodoSQL handles casing.
        create_table_info: meta information about how to create the table

    """
    ev = tracing.Event("create_table_if_not_exists", is_parallel=False)

    # Snowflake import
    try:
        import snowflake.connector  # noqa
    except ImportError:
        raise BodoError(
            "Snowflake Python connector packages not found. "
            "Using 'to_sql' with Snowflake requires snowflake-connector-python. "
            "This can be installed by calling 'conda install -c conda-forge snowflake-connector-python' "
            "or 'pip install snowflake-connector-python'."
        )

    # TODO: handle {table_type}

    # Handle `if_exists` and `table_type`
    if table_type not in ["", "TRANSIENT", "TEMPORARY"]:
        raise ValueError(
            f'"{table_type}" is not valid value for table_type: expected '
            f'"", "TRANSIENT", or "TEMPORARY"'
        )

    if if_exists == "fail":
        create_table_cmd = f"CREATE {table_type} TABLE"
    elif if_exists == "replace":
        create_table_cmd = f"CREATE OR REPLACE {table_type} TABLE"
    elif if_exists == "append":
        create_table_cmd = f"CREATE {table_type} TABLE IF NOT EXISTS"
    else:
        raise ValueError(
            f'"{if_exists}" is not valid for if_exists: expected '
            f'"fail", "replace", or "append"'
        )

    table_comment = None
    column_comments = None

    if create_table_info != None:
        table_comment = create_table_info.table_comment
        column_comments = create_table_info.column_comments

        if create_table_info.table_properties is not None:
            warnings.warn(
                "\nTable properties is not supported in Snowflake. Ignored.\n"
            )

    # Infer schema can return the columns out of order depending on the
    # chunking we do when we upload, so we have to iterate through the
    # dataframe columns to make sure we create the table with its columns
    # in order.
    ev_create_table = tracing.Event("create_table", is_parallel=False)

    # Wrap column names in quotes if they don't match Snowflake's unquoted identifier
    # rules: https://docs.snowflake.com/en/sql-reference/identifiers-syntax
    # BodoSQL matches Snowflake rules so we can always escape column names.
    create_table_col_lst = []
    for col_idx, item in enumerate(sf_schema.items()):
        col_name, typ = item
        if always_escape_col_names or not matches_unquoted_id_rules(col_name):
            col_name = escape_col_name(col_name)
        col_decl = f"{col_name} {typ}"
        if column_comments != None and column_comments[col_idx] != None:
            col_decl = f"{col_decl} COMMENT $${column_comments[col_idx]}$$"
        create_table_col_lst.append(col_decl)
    create_table_columns = ", ".join(create_table_col_lst)
    create_table_sql = f"{create_table_cmd} {location} ({create_table_columns}) "
    if table_comment != None:
        create_table_sql += f" COMMENT = $${table_comment}$$"
    create_table_sql += "/* io.snowflake.create_table_if_not_exists() */"
    cursor.execute(create_table_sql, _is_internal=True)
    ev_create_table.finalize()

    ev.finalize()


def gen_flatten_sql(
    cursor: SnowflakeCursor,
    sf_schema: dict,
    column_datatypes: dict,
    columns: str,
    flatten_table: str,
    location: str,
):  # pragma: no cover
    """Generate the SQL to flatten the table if needed. If flattening is needed, and no flatten_table
    is passed, a temporary table will be created to flatten the data into.
    https://bodo.atlassian.net/wiki/spaces/B/pages/1486815233/Map+Array+Snowflake+Write
    Args:
    cursor: Snowflake connection cursor
    sf_schema (dict): key: dataframe column names, value: dataframe column snowflake datatypes
    column_datatypes (dict): key: dataframe column names, value: dataframe column bodo datatypes
    columns (str): comma-separated column names
    flatten_table (optional(string)): Optionally, specify an existing table to use for flattening
    location (string): desired final location of the data
    """

    # If there are any map arrays they need flattened
    def map_needs_flattened(column_datatype):
        return (
            isinstance(column_datatype, bodo.types.MapArrayType)
            and column_datatype.key_arr_type == bodo.types.string_array_type
        )

    # Group columns on whether they need flattened so we know if we
    # need to flatten
    no_flatten = []
    flatten = []
    for c, data_type in column_datatypes.items():
        if map_needs_flattened(column_datatypes[c]):
            flatten.append(c)
        else:
            no_flatten.append(c)

    needs_flatten = len(flatten) != 0
    if needs_flatten and flatten_table == "":
        temp_schema = {}
        for c, typ in sf_schema.items():
            # Map columns need a variant column in the temp table
            temp_schema[c] = (
                typ
                if not isinstance(column_datatypes[c], bodo.types.MapArrayType)
                else "VARIANT"
            )
        flatten_table = f"bodo_temp_{str(uuid4()).replace('-', '_')}"

        # Create temp table to copy into
        # so the final table can be the correct type
        create_table_handle_exists(
            cursor,
            flatten_table,
            temp_schema,
            "fail",  # This table should never exist
            "TEMPORARY",  # Only persist for this session
            always_escape_col_names=True,
        )

    # Create a subquery to flatten each required column
    subqueries = []
    for i, c in enumerate(flatten):
        select_arg = (",".join(no_flatten) + ", ") if i == 0 and no_flatten else ""
        groups = ("," + ",".join(no_flatten)) if i == 0 and no_flatten else ""
        subqueries.append(
            (
                c,
                f'SELECT rn {c}_rn, {select_arg} OBJECT_AGG("{c}_bodo_flattened".value:key::string, GET("{c}_bodo_flattened".value, \'value\')) {c},'
                f'rn from table_with_rn, LATERAL FLATTEN({c}) "{c}_bodo_flattened" GROUP BY rn, "{c}_bodo_flattened".seq {groups}',
            )
        )

    # Use joins to correlate each column
    subqueries_joined = (
        f"({subqueries[0][1]}) {subqueries[0][0]}_bodo_flattened"
        if len(subqueries)
        else ""
    )
    for i in range(1, len(subqueries)):
        column = subqueries[i][0]
        prev_column = subqueries[i - 1][0]
        subqueries_joined += f" join ({subqueries[i][1]}) {column}_bodo_flattened on {prev_column}_bodo_flattened.{prev_column}_rn = {column}_bodo_flattened.{column}_rn"

    # Figure out where each column comes from
    column_get = (
        [
            (
                f"{c}_bodo_flattened.{c}"
                if map_needs_flattened(column_datatypes[c])
                else f"{flatten[0]}_bodo_flattened.{c}"
            )
            for c in column_datatypes.keys()
        ]
        if needs_flatten
        else []
    )

    flatten_sql = (
        (
            f"INSERT INTO {location} ({columns}) "
            f"WITH table_with_rn as"
            f"  (SELECT ROW_NUMBER() OVER (ORDER BY {columns.split(',')[0]}) rn, * FROM {flatten_table})"
            f"SELECT {','.join(column_get)} FROM {subqueries_joined}"
        )
        if needs_flatten
        else ""
    )
    return flatten_sql, flatten_table


def execute_copy_into(
    cursor: SnowflakeCursor,
    stage_name: str,
    location: str,
    sf_schema,
    column_datatypes,
    synchronous: bool = True,
    stage_dir: str | None = None,
    flatten_table: str = "",
    always_escape_col_names: bool = False,
):  # pragma: no cover
    """Execute a COPY_INTO command from all files in stage to a table location.
    Note: This is intended to be called only from Rank 0.

    Args:
        cursor: Snowflake connection cursor
        stage_name: Name of internal stage containing desired files
        location: Desired table location
        sf_schema (dict): key: dataframe column names, value: dataframe column snowflake datatypes
        column_datatypes (dict): key: dataframe column names, value: dataframe column bodo datatypes
        stage_dir (str or None): Optionally, specify a directory within internal stage
        synchronous (bool): Whether to execute a synchronous COPY INTO command
        flatten_table (optional(string)): Optionally, specify an existing table to use for flattening
        always_escape_col_names (bool): True if we are in BodoSQL table write, which allows always escaping
            column names since BodoSQL handles casing.

    Returns: If synchronous, returns (nsuccess, nchunks, nrows, output) as
        described in `decode_copy_into`. If async, returns COPY INTO Snowflake
        query id as a string.
    """
    ev = tracing.Event("execute_copy_into", is_parallel=False)
    ev.add_attribute("synchronous", synchronous)

    cols_list = []
    # Wrap column names in quotes if they don't match Snowflake's unquoted identifier
    # rules: https://docs.snowflake.com/en/sql-reference/identifiers-syntax
    # BodoSQL matches Snowflake rules so we can always escape column names.
    for col_name in sf_schema.keys():
        if always_escape_col_names or not matches_unquoted_id_rules(col_name):
            col_name = escape_col_name(col_name)
        cols_list.append(f"{col_name}")
    columns = ",".join(cols_list)

    # In Snowflake, all parquet data is stored in a single column, $1,
    # so we must select columns explicitly
    # See (https://docs.snowflake.com/en/user-guide/script-data-load-transform-parquet.html)

    # Binary data: to_binary(col) didn't work as it treats data as HEX
    # BINARY_FORMAT file format option to change this behavior is not supported in
    # copy into parquet to snowflake
    # As a workaround, use ::cast operator and set BINARY_AS_TEXT = False
    # https://docs.snowflake.com/en/user-guide/binary-input-output.html#file-format-option-for-loading-unloading-binary-values
    # https://docs.snowflake.com/en/sql-reference/sql/create-file-format.html#syntax

    binary_mod = {
        c: "::binary" if sf_schema[c] == "BINARY" else "" for c in sf_schema.keys()
    }

    parquet_columns = ",".join([f'$1:"{c}"{binary_mod[c]}' for c in sf_schema.keys()])

    if stage_dir is None:
        stage_name_with_dir = f'@"{stage_name}"'
    else:
        # If the trailing slash is not included here for ADLS stages, the
        # directory itself gets included in the COPY INTO operation and is
        # treated as an empty file, which causes Parquet read to fail on the
        # Snowflake side.
        stage_name_with_dir = f'@"{stage_name}"/{stage_dir}/'

    flatten_sql, flatten_table = gen_flatten_sql(
        cursor, sf_schema, column_datatypes, columns, flatten_table, location
    )

    # Execute copy_into command with files from all ranks
    # TODO: FILE_FROMAT: USE_LOGICAL_TYPE=True for timezone
    copy_into_sql = (
        f"COPY INTO {flatten_table if flatten_table != '' else location} ({columns}) "
        f"FROM (SELECT {parquet_columns} FROM {stage_name_with_dir}) "
        f"FILE_FORMAT=(TYPE=PARQUET COMPRESSION=AUTO BINARY_AS_TEXT=False) "
        f"PURGE=TRUE ON_ERROR={SF_WRITE_COPY_INTO_ON_ERROR} "
        f"/* io.snowflake.execute_copy_into() */"
    )

    if synchronous:
        t0 = time.time()
        copy_results = cursor.execute(copy_into_sql, _is_internal=True).fetchall()  # type: ignore
        sf_exec_time = time.time() - t0
        if bodo.user_logging.get_verbose_level() >= 2:
            bodo.user_logging.log_message(
                "Snowflake Query Submission (Write)",
                "/* io.snowflake.execute_copy_into() */ Snowflake Query ID: "
                + cursor.sfqid
                + "\nSQL Text:\n"
                + copy_into_sql
                + f"\nApproximate Execution Time: {sf_exec_time:.3f}s",
            )
        nsuccess, nchunks, nrows, copy_results = decode_copy_into(copy_results)

        # Print debug output
        ev.add_attribute("copy_into_nsuccess", nsuccess)
        ev.add_attribute("copy_into_nchunks", nchunks)
        ev.add_attribute("copy_into_nrows", nrows)

        if int(os.environ.get("BODO_SF_DEBUG_LEVEL", "0")) >= 1:
            print(f"[Snowflake Write] COPY INTO results:\n{repr(copy_results)}")
            print(f"[Snowflake Write] Total rows: {nrows}")
            print(f"[Snowflake Write] Total files processed: {nchunks}.")
            print(f"[Snowflake Write] Total files successfully processed: {nsuccess}.")
        ev.finalize()
        return nsuccess, nchunks, nrows, copy_results, flatten_sql

    else:
        cursor.execute_async(copy_into_sql, _is_internal=True)  # type: ignore
        ev.finalize()
        return cursor.sfqid, flatten_sql, flatten_table


def retrieve_async_query(cursor: SnowflakeCursor, sfqid: str):  # pragma: no cover
    """Wait for a async query to finish, and return the results.
    If the query fails, this function raises a Snowflake ProgrammingError.
    This function blocks until the query completes / raises an error, and will
    query Snowflake every `SF_WRITE_ASYNC_QUERY_FREQ` seconds.

    Args:
        cursor: Snowflake connection cursor
        sfqid: Snowflake query ID of async query

    Returns:
        result: The query result
    """
    conn = cursor.connection
    while conn.is_still_running(conn.get_query_status_throw_if_error(sfqid)):
        time.sleep(SF_WRITE_ASYNC_QUERY_FREQ)
    cursor.get_results_from_sfqid(sfqid)
    result = cursor.fetchall()
    return result


def decode_copy_into(copy_results):  # pragma: no cover
    """Decode and validate the output of COPY INTO

    Args:
        copy_results: Output of COPY INTO sql command

    Returns (nsuccess, nchunks, nrows, output) where:
        nsuccess (int): Number of chunks successfully copied by the function
        nchunks (int): Number of chunks of data that the function copied
        nrows (int): Number of rows that the function inserted
        output (str): Output of the `COPY INTO <table>` command
    """
    # We have had instances where the output of 'fetchall' may not have tuples with expected
    # lengths or values, hence the error handling.
    nsuccess = 0
    nchunks = 0
    nrows = 0
    for e in copy_results:
        if isinstance(e, tuple):
            if len(e) == 1:
                continue
            nchunks += 1
            if len(e) >= 2 and e[1] == "LOADED":
                nsuccess += 1
            if len(e) >= 4:
                try:
                    nrows += int(e[3])
                except ValueError:  # pragma: no cover
                    pass

    output = repr(copy_results)
    return nsuccess, nchunks, nrows, output


def retrieve_async_copy_into(
    cursor, copy_into_prev_sfqid, file_count, _is_parallel=False
):  # pragma: no cover
    """Retrieve the previous async COPY INTO result, then decode and validate
    the output.

    Args:
        cursor (SnowflakeCursor): Snowflake connection cursor
        copy_into_prev_sfqid (str): Snowflake query ID of async COPY INTO
        file_count (int): Expected file count
        _is_parallel (bool): If True, synchronize errors across ranks

    Returns:
        err (Exception or None): If COPY INTO succeeds, return None.
            If COPY INTO fails, return the exception.
    """
    ev = tracing.Event("retrieve_async_copy_into", is_parallel=False)
    err = None
    nchunks = -1
    nsuccess = -1
    output = ""

    try:
        copy_results = retrieve_async_query(cursor, copy_into_prev_sfqid)
        nsuccess, nchunks, nrows, output = decode_copy_into(copy_results)
        ev.add_attribute("nsuccess", nsuccess)
        ev.add_attribute("nchunks", nchunks)
        ev.add_attribute("nrows", nrows)
        ev.add_attribute("copy_into_output", output)
    except Exception as e:
        err = e

    if err is None and nchunks != file_count:
        rollback_transaction_sql = (
            "ROLLBACK /* io.snowflake.retrieve_async_copy_into() */"
        )
        cursor.execute(rollback_transaction_sql)
        err = BodoError(
            f"Streaming snowflake write failed. Expected COPY INTO to process "
            f"{file_count} files, but only {nchunks} files were found. "
            f"Full COPY INTO result:\n{output}"
        )

    if err is None and nsuccess != nchunks:
        rollback_transaction_sql = (
            "ROLLBACK /* io.snowflake.retrieve_async_copy_into() */"
        )
        cursor.execute(rollback_transaction_sql)
        err = BodoError(
            f"Streaming snowflake write failed. {nchunks} files were loaded, "
            f"but only {nsuccess} were successful. "
            f"Full COPY INTO result:\n{output}"
        )

    ev.finalize()
    return err


# ------------------- Native Distributed Snowflake Write implementation -----------------
# Register opaque type for Snowflake Cursor so it can be shared between
# different sections of jitted code.
# Skip Python type registration if snowflake.connector is not installed,
# since this is an optional dependency.
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import snowflake.connector

        # Update logging information.
        _import_snowflake_connector_logging()

    snowflake_connector_cursor_python_type = snowflake.connector.cursor.SnowflakeCursor
except (ImportError, AttributeError):
    snowflake_connector_cursor_python_type = None

SnowflakeConnectorCursorType, snowflake_connector_cursor_type = install_py_obj_class(
    types_name="snowflake_connector_cursor_type",
    python_type=snowflake_connector_cursor_python_type,
    module=sys.modules[__name__],
    class_name="SnowflakeConnectorCursorType",
    model_name="SnowflakeConnectorCursorModel",
)

# Register opaque type for TemporaryDirectory so it can be shared between
# different sections of jitted code
TemporaryDirectoryType, temporary_directory_type = install_py_obj_class(
    types_name="temporary_directory_type",
    python_type=TemporaryDirectory,
    module=sys.modules[__name__],
    class_name="TemporaryDirectoryType",
    model_name="TemporaryDirectoryModel",
)


def get_snowflake_stage_info(
    cursor: SnowflakeCursor,
    stage_name: str,
    tmp_folder: TemporaryDirectory,
) -> dict:  # pragma: no cover
    """Get parquet path and credentials for a snowflake internal stage.
    This works by using `_execute_helper` to issue a dummy upload query

    Args
        cursor: Snowflake connection cursor
        stage_name: Stage name to query information about
        tmp_folder: A TemporaryDirectory() object
            representing a temporary directory on disk to store files
            prior to an upload

    Returns
        stage_info: Dictionary of snowflake stage info
    """
    ev = tracing.Event("get_snowflake_stage_info", is_parallel=False)

    # Create a unique filepath for dummy upload query with quotes/backslashes escaped
    query_path = os.path.join(tmp_folder.name, f"get_credentials_{uuid4()}.parquet")
    # To escape backslashes, we want to replace ( \ ) with ( \\ ), which can
    # be written as the string literals ( \\ ) and ( \\\\ ).
    # To escape quotes, we want to replace ( ' ) with ( \' ), which can
    # be written as the string literals ( ' ) and ( \\' ).
    query_path = query_path.replace("\\", "\\\\").replace("'", "\\'")

    # Run `_execute_helper` to get stage info dict from Snowflake
    upload_sql = (
        f"PUT 'file://{query_path}' @\"{stage_name}\" AUTO_COMPRESS=FALSE "
        f"/* io.snowflake.get_snowflake_stage_info() */"
    )
    stage_info = cursor._execute_helper(upload_sql, is_internal=True)

    ev.finalize()
    return stage_info


def connect_and_get_upload_info(conn_str: str):  # pragma: no cover
    """On rank 0, connect to Snowflake, create an internal stage, and issue
    an upload command to get parquet path and internal stage credentials.
    If the internal stage type is not supported or SF_WRITE_UPLOAD_USING_PUT
    is True, use the PUT implementation by connecting to Snowflake on all ranks.
    This function exists to be executed within objmode from `DataFrame.to_sql()`

    Note that we set the session timezone to UTC. This is because of a gap in
    Snowflake's handling of timestamp types during parquet ingestion
    (https://github.com/snowflakedb/snowflake-connector-python/issues/1687).
    See comment in the code for more details.
    This is safe since this Snowflake Connection object / cursor is only used
    in the Snowflake write case. In the future, if we want to use a single
    Snowflake connection for all commands in a SQL query (e.g. transactions),
    this could be unsafe and needs to be revisited.

    Args
        conn_str: Snowflake connection URL string

    Returns: (cursor, tmp_folder, stage_name, parquet_path, upload_using_snowflake_put, old_creds) where
        cursor (snowflake.connector.cursor): Snowflake connection cursor
        tmp_folder (TemporaryDirectory): A TemporaryDirectory() object
            representing a temporary directory on disk to store files
            prior to an upload
        stage_name (str): Name of created internal stage
        parquet_path (str): Parquet path of internal stage, either an S3/ADLS
            URI or a local path in the case of upload using PUT, with trailing slash
        upload_using_snowflake_put (bool): An updated boolean flag for whether
            we are using the PUT command in objmode to upload files. This is
            set to True if we don't support the stage type returned by Snowflake.
        old_creds (Dict(str, str)): Old environment variables that were
            overwritten to update credentials for uploading to stage
    """
    ev = tracing.Event("connect_and_get_upload_info")

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()

    # Create a temporary directory on every rank
    tmp_folder = TemporaryDirectory()

    # On rank 0, create named internal stage and get stage info dict
    cursor = None  # Forward declaration
    stage_name = ""  # Forward declaration
    parquet_path = ""  # Forward declaration
    upload_creds = {}  # Forward declaration
    old_creds = {}  # Forward declaration

    err = None  # Forward declaration
    if my_rank == 0:
        try:
            # Connect to snowflake
            conn = snowflake_connect(conn_str)
            cursor = conn.cursor()
            # Temporary solution until `use_logical_type` is used
            # (https://github.com/snowflakedb/snowflake-connector-python/issues/1687).
            # Parquet converts Timezone-aware data to UTC (default behavior Bodo already has).
            # Set Snowflake TIMEZONE Session to UTC before COPY INTO
            # to avoid Snowflake from doing another conversion to UTC.
            # Note that our parquet files do set `isAdjustedToUTC` to true for timestamp
            # columns. It's just that Snowflake is not able to use this logical
            # type information yet.
            # See comment in [BSE-1476]
            change_timezone_session_sql = "ALTER SESSION SET TIMEZONE = 'UTC'"
            cursor.execute(change_timezone_session_sql).fetchall()
            # Avoid creating a temp stage at all in case of SF_WRITE_UPLOAD_USING_PUT
            is_temporary = not SF_WRITE_UPLOAD_USING_PUT
            stage_name = create_internal_stage(cursor, is_temporary=is_temporary)

            if SF_WRITE_UPLOAD_USING_PUT:
                # With this config option set, always upload using snowflake PUT.
                # An empty parquet path denotes fallback to objmode PUT below
                parquet_path = ""
            else:
                # Parse stage info dict for parquet path and credentials
                stage_info = get_snowflake_stage_info(cursor, stage_name, tmp_folder)
                upload_info = stage_info["data"]["uploadInfo"]

                location_type = upload_info.get("locationType", "UNKNOWN")
                fallback_to_put = False

                if location_type == "S3":
                    # Parquet path format: s3://<bucket_name>/<key_name>
                    # E.g. s3://sfc-va2-ds1-9-customer-stage/b9zr-s-v2st3620/stages/547e65a7-fa2c-491b-98c3-6e4313db7741/
                    # See https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-bucket-intro.html#accessing-a-bucket-using-S3-format
                    bucket_name, _, path = upload_info["location"].partition("/")
                    path = path.rstrip("/")

                    parquet_path = f"s3://{bucket_name}/{path}/"
                    upload_creds = {
                        "AWS_ACCESS_KEY_ID": upload_info["creds"]["AWS_KEY_ID"],
                        "AWS_SECRET_ACCESS_KEY": upload_info["creds"]["AWS_SECRET_KEY"],
                        "AWS_SESSION_TOKEN": upload_info["creds"]["AWS_TOKEN"],
                        "AWS_DEFAULT_REGION": upload_info["region"],
                    }
                elif location_type == "AZURE":
                    # Upload path format: abfs[s]://<file_system>@<account_name>.dfs.core.windows.net/<path>/<file_name>
                    # E.g. abfs://stageszz05dc579c-e473-4aa2-b8a3-62a1ae425a11@qiavr8sfcb1stg.dfs.core.windows.net/<file_name>
                    # For URI syntax, see https://docs.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-introduction-abfs-uri#uri-syntax
                    container_name, _, path = upload_info["location"].partition("/")
                    path = path.rstrip("/")

                    account_name = upload_info["storageAccount"]
                    sas_token = upload_info["creds"]["AZURE_SAS_TOKEN"]

                    if len(path) == 0:
                        parquet_path = f"abfs://{container_name}@{account_name}.dfs.core.windows.net/{sas_token}"
                    else:
                        parquet_path = f"abfs://{container_name}@{account_name}.dfs.core.windows.net/{path}/{sas_token}"

                else:
                    # Unsupported internal stage location. This code falls back to objmode upload
                    fallback_to_put = True
                    warnings.warn(
                        BodoWarning(
                            f"Direct upload to stage is not supported for internal stage "
                            f"type '{location_type}'. Falling back to PUT "
                            f"command for upload."
                        )
                    )

                if fallback_to_put:
                    # If falling back to PUT method, drop this stage and create a non-temporary
                    # stage instead.
                    drop_internal_stage(cursor, stage_name)
                    stage_name = create_internal_stage(cursor, is_temporary=False)

        except Exception as e:
            err = RuntimeError(str(e))
            if int(os.environ.get("BODO_SF_DEBUG_LEVEL", "0")) >= 1:
                print("".join(traceback.format_exception(None, e, e.__traceback__)))

    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err

    parquet_path = comm.bcast(parquet_path)

    if parquet_path == "":
        # Falling back to PUT for upload. The internal stage type could be
        # unsupported, or the `upload_using_snowflake_put` flag could be set to True.
        upload_using_snowflake_put = True
        parquet_path = tmp_folder.name + "/"

        # Objmode PUT requires a Snowflake connection on all ranks, not just rank 0
        if my_rank != 0:
            # Since we already connected to Snowflake successfully on rank 0,
            # unlikely we'll have an exception here.
            conn = snowflake_connect(conn_str)
            cursor = conn.cursor()

    else:
        upload_using_snowflake_put = False

        # On all ranks, update environment variables with internal stage credentials
        upload_creds = comm.bcast(upload_creds)
        old_creds = update_env_vars(upload_creds)

    stage_name = comm.bcast(stage_name)

    ev.finalize()
    return (
        cursor,
        tmp_folder,
        stage_name,
        parquet_path,
        upload_using_snowflake_put,
        old_creds,
    )


def create_table_copy_into(
    cursor: SnowflakeCursor,
    stage_name: str,
    location: str,
    sf_schema: dict,
    column_datatypes: dict,
    if_exists: str,
    table_type: str,
    num_files_uploaded: int,
    old_creds,
    tmp_folder: TemporaryDirectory,
):  # pragma: no cover
    """
    Auto-create a new table if needed, execute COPY_INTO, and clean up
    created internal stage, and restore old environment variables.
    This function exists to be executed within objmode from `DataFrame.to_sql()`

    Args
        cursor: Snowflake connection cursor
        stage_name: Name of internal stage containing files to copy_into
        location: Destination table location
        sf_schema (dict): key: dataframe column names, value: dataframe column snowflake datatypes
        column_datatypes (dict): key: dataframe column names, value: dataframe column bodo datatypes
        if_exists: Action to take if table already exists:
            "fail": If table exists, raise a ValueError. Create if does not exist
            "replace": If table exists, drop it, recreate it, and insert data.
                Create if does not exist
            "append": If table exists, insert data. Create if does not exist
        table_type: Type of table to create. Must be one of "", "TRANSIENT", or "TEMPORARY"
        num_files_uploaded: Number of files that were uploaded to the stage. We use this
            to validate that the COPY INTO went through successfully. Also, in case
            this is 0, we skip the COPY INTO step.
        old_creds (Dict(str, str or None)): Old environment variables to restore.
            Previously overwritten to update credentials for uploading to stage
        tmp_folder: TemporaryDirectory object to clean up
    """
    ev = tracing.Event("create_table_copy_into", is_parallel=False)
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()

    # On rank 0, create a new table if needed, then execute COPY_INTO
    err = None  # Forward declaration
    if my_rank == 0:
        try:
            begin_transaction_sql = "BEGIN /* io.snowflake.create_table_copy_into() */"
            cursor.execute(begin_transaction_sql)

            # Table should be created even if the dataframe is empty.
            create_table_handle_exists(
                cursor,
                location,
                sf_schema,
                if_exists,
                table_type,
            )
            # No point of running COPY INTO if there are no files.
            if num_files_uploaded > 0:
                nsuccess, nchunks, nrows, copy_results, flatten_sql = execute_copy_into(
                    cursor,
                    stage_name,
                    location,
                    sf_schema,
                    column_datatypes,
                    synchronous=True,
                )

                # Validate copy into results
                if nchunks != num_files_uploaded:
                    rollback_transaction_sql = (
                        "ROLLBACK /* io.snowflake.create_table_copy_into() */"
                    )
                    cursor.execute(rollback_transaction_sql)
                    raise BodoError(
                        f"Snowflake write failed. Expected COPY INTO to have processed "
                        f"{num_files_uploaded} files, but only {nchunks} files were found. "
                        f"Full COPY INTO result:\n{copy_results}"
                    )

                if nsuccess != nchunks:
                    rollback_transaction_sql = (
                        "ROLLBACK /* io.snowflake.create_table_copy_into() */"
                    )
                    cursor.execute(rollback_transaction_sql)
                    raise BodoError(
                        f"Snowflake write failed. {nchunks} files were loaded, but only "
                        f"{nsuccess} were successful. "
                        f"Full COPY INTO result:\n{copy_results}"
                    )

                # Execute flatten query if needed
                if len(flatten_sql) != 0:
                    cursor.execute(flatten_sql)

            commit_transaction_sql = (
                "COMMIT /* io.snowflake.create_table_copy_into() */"
            )
            cursor.execute(commit_transaction_sql)

            drop_internal_stage(cursor, stage_name)

            cursor.close()

        except Exception as e:
            err = RuntimeError(str(e))
            if int(os.environ.get("BODO_SF_DEBUG_LEVEL", "0")) >= 1:
                print("".join(traceback.format_exception(None, e, e.__traceback__)))

    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err

    # Put back the environment variables
    update_env_vars(old_creds)

    # Cleanup the folder that was created to store parquet chunks for upload
    tmp_folder.cleanup()

    ev.finalize()
