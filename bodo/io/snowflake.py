import os
import re
import sys
import traceback
import warnings
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Tuple
from urllib.parse import parse_qsl, urlparse
from uuid import uuid4

import pyarrow as pa
from mpi4py import MPI
from numba.core import types

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.io.helpers import (
    ExceptionPropagatingThread,
    _get_numba_typ_from_pa_typ,
    update_env_vars,
    update_file_contents,
)
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.dict_arr_ext import dict_str_arr_type
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.utils import tracing
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.typing import BodoError, BodoWarning, is_str_arr_type

# Imports for typechecking
if TYPE_CHECKING:  # pragma: no cover
    from snowflake.connector import SnowflakeConnection
    from snowflake.connector.cursor import ResultMetadata, SnowflakeCursor
    from snowflake.connector.result_batch import JSONResultBatch, ResultBatch

# How long the schema / typeof probe query should run for in the worst case.
# This is to guard against increasing compilation time prohibitively in case there are
# issues with Snowflake, the data, etc.
SF_READ_SCHEMA_PROBE_TIMEOUT = 5

# Whether to do a probe query to determine whether string columns should be
# dictionary-encoded. This doesn't effect the _bodo_read_as_dict argument.
SF_READ_AUTO_DICT_ENCODE_ENABLED = True

# A configurable variable by which we determine whether to dictionary-encode
# a string column.
# Encode if num of unique elem / num of total rows <= SF_READ_DICT_ENCODE_CRITERION
SF_READ_DICT_ENCODE_CRITERION = 0.5

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

# Mapping of the Snowflake field types to the pyarrow types taken
# from the snowflake connector. These are not fully accurate and don't match
# the arrow types. However, they can be used when the returned data is empty.
# https://github.com/snowflakedb/snowflake-connector-python/blob/dcf10e8c7ce13a5288104b28329d3c9e8ffffc5a/src/snowflake/connector/constants.py#L35
# https://docs.snowflake.com/en/user-guide/python-connector-api.html#label-python-connector-type-codes
SCALE_TO_UNIT_PRECISION: Dict[int, Literal["s", "ms", "us", "ns"]] = {
    0: "s",
    3: "ms",
    6: "us",
    9: "ns",
}
TYPE_CODE_TO_ARROW_TYPE: List[Callable[["ResultMetadata", str], pa.DataType]] = [
    # Number / Int - Always Signed
    lambda m, _: pa.int64()
    if m.scale == 0
    else (pa.float64() if m.scale < 18 else pa.decimal128(m.precision, m.scale)),
    # Float / Double
    lambda _, __: pa.float64(),
    # String
    lambda _, __: pa.string(),
    # Dates - Snowflake stores in days (aka 32-bit)
    lambda _, __: pa.date32(),
    # Timestamp - Seems to be unused?
    lambda _, __: pa.time64("ns"),
    # Variant / Union Type
    lambda _, __: pa.string(),
    # Timestamp stored in UTC - TIMESTAMP_LTZ
    lambda m, tz: pa.timestamp(SCALE_TO_UNIT_PRECISION[m.scale], tz=tz),
    # Timestamp with a timezone offset per item - TIMESTAMP_TZ
    lambda m, tz: pa.timestamp(SCALE_TO_UNIT_PRECISION[m.scale], tz=tz),
    # Timestamp without a timezone - TIMESTAMP_NTZ
    lambda m, _: pa.timestamp(SCALE_TO_UNIT_PRECISION[m.scale]),
    # Object / Struct - Connector doesn't support pa.struct
    lambda _, __: pa.string(),
    # Array - Connector doesn't support pa.list
    lambda _, __: pa.string(),
    # Binary
    lambda _, __: pa.binary(),
    # Time
    lambda m, _: (
        {0: pa.time32("s"), 3: pa.time32("ms"), 6: pa.time64("us"), 9: pa.time64("ns")}
    )[m.scale],
    # Boolean
    lambda _, __: pa.bool_(),
    # Geographic - No Core Arrow Equivalent
    lambda _, __: pa.string(),
]
INT_BITSIZE_TO_ARROW_DATATYPE = {
    1: pa.int8(),
    2: pa.int16(),
    4: pa.int32(),
    8: pa.int64(),
    16: pa.decimal128(38, 0),
}
STRING_TYPE_CODE = 2


def gen_snowflake_schema(column_names, column_datatypes):  # pragma: no cover
    """Generate a dictionary where column is key and
    its corresponding bodo->snowflake datatypes is value

    Args:
        column_names (array-like): Array of DataFrame column names
        column_datatypes (array-like): Array of DataFrame column datatypes

    Returns:
        sf_schema (dict): {col_name : snowflake_datatype}
    Raises BodoError for unsupported datatypes when writing to snowflake.
    """
    sf_schema = {}
    for col_name, col_type in zip(column_names, column_datatypes):
        if col_name == "":
            raise BodoError("Column name cannot be empty when writing to Snowflake.")
        # TODO: differentiate between timezone aware or not types.
        # [BE-3587] need specific tz for each column type.
        if isinstance(col_type, bodo.DatetimeArrayType) or (
            col_type == bodo.datetime_datetime_type
        ):
            sf_schema[col_name] = "TIMESTAMP_NTZ"
        elif col_type == bodo.datetime_date_array_type:
            sf_schema[col_name] = "DATE"
        elif isinstance(col_type, bodo.TimeArrayType):
            if col_type.precision in [0, 3, 6]:
                precision = col_type.precision
            elif col_type.precision == 9:
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
            sf_schema[col_name] = f"TIME({precision})"
        elif isinstance(col_type, types.Array):
            numpy_type = col_type.dtype.name
            if numpy_type.startswith("datetime"):
                sf_schema[col_name] = "DATETIME"
            # NOTE: Bodo matches Pandas behavior
            # and prints same warning and save it as a number.
            if numpy_type.startswith("timedelta"):
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
            elif numpy_type.startswith(("float")):
                sf_schema[col_name] = "REAL"
        elif is_str_arr_type(col_type):
            sf_schema[col_name] = "TEXT"
        elif col_type == bodo.binary_array_type:
            sf_schema[col_name] = "BINARY"
        elif col_type == bodo.boolean_array_type:
            sf_schema[col_name] = "BOOLEAN"
        # TODO: differentiate between unsigned vs. signed, 8, 16, 32, 64
        elif isinstance(col_type, bodo.IntegerArrayType):
            sf_schema[col_name] = "NUMBER(38, 0)"
        elif isinstance(col_type, bodo.FloatingArrayType):
            sf_schema[col_name] = "REAL"
        elif isinstance(col_type, bodo.DecimalArrayType):
            sf_schema[col_name] = "NUMBER(38, 18)"
        elif isinstance(col_type, (ArrayItemArrayType, StructArrayType)):
            # based on testing with infer_schema
            sf_schema[col_name] = "VARIANT"
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

# SF_WRITE_OVERLAP_UPLOAD (bool):
# If True, attempt to overlap writing dataframe to Parquet files and
#     uploading Parquet files to Snowflake internal stage using Python
#     threads on each rank. This speeds up the upload process by
#     overlapping compute-heavy and I/O-heavy steps.
# If False, perform these steps in sequence without spawning threads.
SF_WRITE_OVERLAP_UPLOAD = True

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

# SF_WRITE_UPLOAD_USING_PUT (bool):
# If True, `to_sql` saves the dataframe to Parquet files in a local
#     temporary directory, then uses Snowflake's PUT command to upload
#     to internal stage (https://docs.snowflake.com/en/sql-reference/sql/put.html)
#     This method supports all Snowflake stage types but may be slower.
# If False, `to_sql` directly uploads Parquet files to S3/ADLS/GCS using Bodo's
#     internal filesystem-write infrastructure. This method is faster but does
#     not support Azure and GCS-backed Snowflake accounts.
SF_WRITE_UPLOAD_USING_PUT = False


# Content to put in core-site.xml for Snowflake write. For ADLS backed stages,
# Snowflake gives us a SAS token for access to the stage. Unfortunately,
# SAS tokens cannot be directly provided in core-site, and instead require
# a SASTokenProvider implementation. This core-site specifies our own
# SASTokenProvider class (BodoSASTokenProvider) as the implementation.
# This implementation simply reads the token from a file
# (SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION) and returns it. Note that
# this is meant to be a constant, not something that  users should
# need to modify.
SF_AZURE_WRITE_HDFS_CORE_SITE = """<configuration>
  <property>
    <name>fs.azure.account.auth.type</name>
    <value>SAS</value>
  </property>
  <property>
    <name>fs.azure.sas.token.provider.type</name>
    <value>org.bodo.azurefs.sas.BodoSASTokenProvider</value>
  </property>
  <property>
    <name>fs.abfs.impl</name>
    <value>org.apache.hadoop.fs.azurebfs.AzureBlobFileSystem</value>
  </property>
</configuration>
"""

# Temporary location to write the SAS token to. This will
# be read by BodoSASTokenProvider.
# Keep this in sync with the location in BodoSASTokenProvider.java
SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION = os.path.join(
    bodo.HDFS_CORE_SITE_LOC_DIR.name, "sas_token.txt"
)


def execute_query(
    cursor: "SnowflakeCursor",
    query: str,
    timeout: Optional[int],
) -> Optional["SnowflakeCursor"]:  # pragma: no cover
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
            raise


def escape_col_name(col_name: str) -> str:
    """Helper Function to Escape Snowflake Column Names"""
    return '"{}"'.format(col_name.replace('"', '""'))


def snowflake_connect(
    conn_str: str, is_parallel: bool = False
) -> "SnowflakeConnection":  # pragma: no cover
    """
    From Snowflake connection URL, connect to Snowflake.

    Args:
        conn_str: Snowflake connection URL in the following format:
            snowflake://<user_login_name>:<password>@<account_identifier>/<database_name>/<schema_name>?warehouse=<warehouse_name>&role=<role_name>
            Required arguments include <user_login_name>, <password>, and
            <account_identifier>. Optional arguments include <database_name>,
            <schema_name>, <warehouse_name>, and <role_name>.
            Do not include the `snowflakecomputing.com` domain name as part of
            your account identifier. Snowflake automatically appends the domain
            name to your account identifier to create the required connection.
            (https://docs.snowflake.com/en/user-guide/sqlalchemy.html#connection-parameters)
        is_parallel: True if this function being is called from all
            ranks, and False otherwise

    Returns
        conn: Snowflake Connection object
    """
    ev = tracing.Event("snowflake_connect", is_parallel=is_parallel)

    u = urlparse(conn_str)
    params = {}
    if u.username:
        params["user"] = u.username
    if u.password:
        params["password"] = u.password
    if u.hostname:
        params["account"] = u.hostname
    if u.port:
        params["port"] = u.port
    if u.path:
        # path contains "database_name/schema_name"
        path = u.path
        if path.startswith("/"):
            path = path[1:]
        parts = path.split("/")
        if len(parts) == 2:
            database, schema = parts
        elif len(parts) == 1:  # pragma: no cover
            database = parts[0]
            schema = None
        else:  # pragma: no cover
            raise BodoError(
                f"Unexpected Snowflake connection string {conn_str}. Path is expected to contain database name and possibly schema"
            )
        params["database"] = database
        if schema:
            params["schema"] = schema
    if u.query:
        # query contains warehouse_name and role_name
        for key, val in parse_qsl(u.query):
            params[key] = val
            if key == "session_parameters":
                # Snowflake connector appends to session_parameters and
                # assumes it is a dictionary if provided. This is an existing
                # bug in SqlAlchemy/SnowflakeSqlAlchemy
                import json

                params[key] = json.loads(val)

    # Set a short login timeout so people don't have to wait the default
    # 60 seconds to find out they added the wrong credentials.
    params["login_timeout"] = 5

    try:
        import snowflake.connector
    except ImportError:
        raise BodoError(
            "Snowflake Python connector packages not found. "
            "Using 'to_sql' with Snowflake requires snowflake-connector-python. "
            "This can be installed by calling 'conda install -c conda-forge snowflake-connector-python' "
            "or 'pip install snowflake-connector-python'."
        )
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
                f"Performance Warning: The Snowflake warehouse and Bodo platform are on different cloud providers. "
                + f"The Snowflake warehouse is located on {sf_cloud_provider}, but the Bodo cluster is located on {platform_cloud_provider}. "
                + "For best performance we recommend using your cluster and Snowflake account in the same region with the same cloud provider."
            )
            warnings.warn(warning)
        elif platform_region_str != sf_cloud_region:
            warning = BodoWarning(
                f"Performance Warning: The Snowflake warehouse and Bodo platform are in different cloud regions. "
                + f"The Snowflake warehouse is located in {sf_cloud_region}, but the Bodo cluster is located in {platform_region_str}. "
                + "For best performance we recommend using your cluster and Snowflake account in the same region with the same cloud provider."
            )
            warnings.warn(warning)
    ev.finalize()
    return conn


def get_schema_from_metadata(
    cursor: "SnowflakeCursor",
    sql_query: str,
    is_select_query: bool,
    is_table_input: bool,
) -> Tuple[
    List[pa.Field], List, List[bool], List[int], List[pa.DataType]
]:  # pragma: no cover
    """
    Determine the Arrow schema and Bodo types of the query output
    The approach is described in a Confluence Doc:
    https://bodo.atlassian.net/wiki/spaces/B/pages/1238433836/Snowflake+Read+Table+Schema+Inference
    This function executes independently on ranks.

    Args:
        cursor: Snowflake Cursor to Perform Operations in
        sql_query: Base SQL Query Operation
        is_select_query: sql_query is a SELECT query

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

    arrow_fields: List[pa.Field] = []  # Equivalent PyArrow Fields
    col_names_to_check: List[str] = []  # Columns to get typeof metadata for
    check_dict_encoding: List[
        bool
    ] = []  # Should we check dictionary encoding for this type.
    col_idxs_to_check: List[int] = []  # Index of columns to get typeof metadata
    for i, field_meta in enumerate(query_field_metadata):
        dtype = TYPE_CODE_TO_ARROW_TYPE[field_meta.type_code](field_meta, tz)
        arrow_fields.append(pa.field(field_meta.name, dtype, field_meta.is_nullable))
        # Only check dictionary encoding for STRING columns, not other columns
        # that we load as strings.
        check_dict_encoding.append(field_meta.type_code == STRING_TYPE_CODE)
        if pa.types.is_int64(dtype):
            col_names_to_check.append(field_meta.name)
            col_idxs_to_check.append(i)

    # For any NUMBER columns, fetch SYSTEM$TYPEOF metadata to determine
    # the smallest viable integer type (number of bytes)
    if is_select_query and len(col_names_to_check) != 0:
        schema_probe_query = (
            "SELECT "
            + ", ".join(
                f"SYSTEM$TYPEOF({escape_col_name(x)})" for x in col_names_to_check
            )
            + f" FROM ({sql_query}) LIMIT 1"
        )

        probe_res = execute_query(
            cursor, schema_probe_query, timeout=SF_READ_SCHEMA_PROBE_TIMEOUT
        )
        if (
            probe_res is not None
            and (typing_table := probe_res.fetch_arrow_all()) is not None
        ):
            # Note, this assumes that the output metadata columns are in the
            # same order as the columns we checked in the probe query
            for i, (full_name, typing_info) in enumerate(
                typing_table.to_pylist()[0].items()
            ):
                orig_col_name = col_names_to_check[i]
                exp_col_names = (
                    f"SYSTEM$TYPEOF({escape_col_name(orig_col_name)})",
                    f"SYSTEM$TYPEOF({escape_col_name(orig_col_name.upper())})",
                )
                assert (
                    full_name in exp_col_names
                ), "Output of Snowflake Schema Probe Query Uses Unexpected Column Names"

                idx = col_idxs_to_check[i]
                # Parse output NUMBER(__,_)[SBx] to get the byte width x
                byte_size = int(
                    re.search("NUMBER\(\d+,\d+\)\[SB(\d+)\]", typing_info).group(1)
                )
                out_type = INT_BITSIZE_TO_ARROW_DATATYPE[byte_size]
                arrow_fields[idx] = arrow_fields[idx].with_type(out_type)

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
        check_dict_encoding,
        unsupported_columns,
        unsupported_arrow_types,
    )


def _get_table_row_count(cursor, table_name):
    """get total number of rows for a Snowflake table. Returns None if input is not a
    table or probe query failed.

    Args:
        cursor (SnowflakeCursor): Snowflake connector connection cursor object
        table_name (str): table name

    Returns:
        optional(int): number of rows or None if failed
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
):
    """Detects Snowflake columns that need to be dictionary-encoded using a query that
    gets approximate data cardinalities.

    Args:
        sql_query (str): read query or Snowflake table name
        query_args (list(str)): probe query arguments, e.g. ['approx_count_distinct(A)',
            'approx_count_distinct(B)']
        string_col_ind (list(int)): index of string columns in col_types
        undetermined_str_cols (iterable(str)): column names of string columns that need
            dict-encoding probe (not manually specified)
        cursor (SnowflakeCursor): Snowflake connector connection cursor object
        col_types (list(types.Type)): read data types to update with dict-encoding info
        is_table_input (bool): read query is a table name

    Returns:
        Optional[Tuple[int, List[str]]]: debug info if the probe query timed out
    """

    # Determine if the string columns are dictionary encoded
    dict_encode_timeout_info: Optional[Tuple[int, List[str]]] = None

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
        check_res = execute_query(
            cursor,
            # Note we need both like and starts with because in this context
            # like is case-insensitive but starts with is case-sensitive. Since they
            # are exactly the same this will only match the exact query.
            # See https://bodo.atlassian.net/browse/BSE-277 for why this is necessary.
            f"show tables like '{sql_query}' starts with '{sql_query}'",
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
        uniqueness = [
            cardinality_data[i][0].as_py() / max(total_rows, 1)
            for i in range(1, len(query_args) + 1)
        ]
        # filter the string col indices based on the criterion
        col_inds_to_convert = filter(
            lambda x: x[0] <= SF_READ_DICT_ENCODE_CRITERION,
            zip(uniqueness, string_col_ind),
        )
        for _, ind in col_inds_to_convert:
            col_types[ind] = dict_str_arr_type

    return dict_encode_timeout_info


def get_schema(
    conn_str: str,
    sql_query: str,
    is_select_query: bool,
    is_table_input: bool,
    _bodo_read_as_dict: Optional[List[str]],
):  # pragma: no cover
    conn = snowflake_connect(conn_str)
    cursor = conn.cursor()

    (
        pa_fields,
        col_types,
        check_dict_encoding,
        unsupported_columns,
        unsupported_arrow_types,
    ) = get_schema_from_metadata(cursor, sql_query, is_select_query, is_table_input)

    str_as_dict_cols = _bodo_read_as_dict if _bodo_read_as_dict else []
    str_col_name_to_ind = {}
    for i, check_dict_encoding in enumerate(check_dict_encoding):
        if check_dict_encoding:
            str_col_name_to_ind[pa_fields[i].name] = i

    # Map the snowflake original column name to the name that
    # is used from Python. This is used for comparing with
    # _bodo_read_as_dict which will use Python's convention.
    snowflake_case_map = {
        name.lower() if name.isupper() else name: name
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
        # Always quote column names for correctness
        query_args.append(f'approx_count_distinct("{snowflake_case_map[name]}")')
        string_col_ind.append(str_col_name_to_ind[snowflake_case_map[name]])

    # Determine if the string columns are dictionary encoded
    dict_encode_timeout_info: Optional[Tuple[int, List[str]]] = None

    if len(query_args) != 0 and SF_READ_AUTO_DICT_ENCODE_ENABLED:
        dict_encode_timeout_info = _detect_column_dict_encoding(
            sql_query,
            query_args,
            string_col_ind,
            undetermined_str_cols,
            cursor,
            col_types,
            is_table_input,
        )

    # Ensure column name case matches Pandas/sqlalchemy. See:
    # https://github.com/snowflakedb/snowflake-sqlalchemy#object-name-case-handling
    # If a name is returned as all uppercase by the Snowflake connector
    # it means it is case insensitive or it was inserted as all
    # uppercase with double quotes. In both of these situations
    # pd.read_sql() returns the name with all lower case
    final_colnames: List[str] = []
    converted_colnames = set()
    for x in pa_fields:
        if x.name.isupper():
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
        dict_encode_timeout_info,
    )


class SnowflakeDataset(object):
    """Store dataset info in the way expected by Arrow reader in C++."""

    def __init__(
        self, batches: List["ResultBatch"], schema, conn: "SnowflakeConnection"
    ):
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


class FakeArrowJSONResultBatch:
    """
    Results Batch used to return a JSONResult in arrow format while
    conforming to the same APIS as ArrowResultBatch
    """

    def __init__(self, json_batch: "JSONResultBatch", schema: pa.Schema) -> None:
        self._json_batch = json_batch
        self._schema = schema

    @property
    def rowcount(self):
        return self._json_batch.rowcount

    def to_arrow(self, _: Optional["SnowflakeConnection"] = None) -> pa.Table:
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


def get_dataset(
    query: str,
    conn_str: str,
    schema: pa.Schema,
    only_fetch_length: bool = False,
    is_select_query: bool = True,
    is_parallel: bool = True,
    is_independent: bool = False,
) -> Tuple[SnowflakeDataset, int]:  # pragma: no cover
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
    assert not (
        only_fetch_length and not is_select_query
    ), "The only length optimization can only be run with select queries"

    # Data cannot be distributed if each rank is independent
    assert not (
        is_parallel and is_independent
    ), "Snowflake get_dataset: is_parallel and is_independent cannot be True at the same time"

    # Snowflake import
    try:
        import snowflake.connector  # noqa
        from snowflake.connector.result_batch import (
            ArrowResultBatch,
            JSONResultBatch,
        )
    except ImportError:
        raise BodoError(
            "Snowflake Python connector packages not found. "
            "Fetching data from Snowflake requires snowflake-connector-python. "
            "This can be installed by calling 'conda install -c conda-forge snowflake-connector-python' "
            "or 'pip install snowflake-connector-python'."
        )

    ev = tracing.Event("get_snowflake_dataset", is_parallel=is_parallel)

    comm = MPI.COMM_WORLD

    # connect to Snowflake. This is the same connection that will be used
    # to read data.
    # We only trace on rank 0 (is_parallel=False) because we want 0 to start
    # executing the queries as soon as possible (don't sync event)
    conn = snowflake_connect(conn_str)
    # Number of rows loaded. This is only used if we are loading
    # 0 columns
    num_rows = -1
    batches = []

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
            cur = conn.cursor()
            ev_query = tracing.Event("execute_length_query", is_parallel=False)
            ev_query.add_attribute("query", query)
            cur.execute(query)
            # We are just loading a single row of data so we can just load
            # all of the data.
            arrow_data = cur.fetch_arrow_all()
            num_rows = arrow_data[0][0].as_py()  # type: ignore
            cur.close()
            ev_query.finalize()

        # If the ranks are not independent from each other, broadcast num_rows
        if not is_independent:
            num_rows = comm.bcast(num_rows)
    else:
        # We need to actually submit a Snowflake query
        if bodo.get_rank() == 0 or is_independent:
            # Execute query
            ev_query = tracing.Event("execute_query", is_parallel=False)
            ev_query.add_attribute("query", query)
            cur = conn.cursor()
            cur.execute(query)
            ev_query.finalize()

            # Fetch the total number of rows that will be loaded globally
            num_rows: int = cur.rowcount  # type: ignore

            # Get the list of result batches (this doesn't load data).
            batches: "List[ResultBatch]" = cur.get_result_batches()  # type: ignore
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

            cur.close()

        # If the ranks are not independent from each other, broadcast the data
        if not is_independent:
            num_rows, batches, schema = comm.bcast((num_rows, batches, schema))

    ds = SnowflakeDataset(batches, schema, conn)
    ev.finalize()
    return ds, num_rows


# --------------------------- snowflake_write helper functions ----------------------------
def create_internal_stage(cursor: "SnowflakeCursor", is_temporary: bool = False) -> str:
    """Create an internal stage within Snowflake. If `is_temporary=False`,
    the named stage must be dropped manualy in `drop_internal_stage()`

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


def drop_internal_stage(cursor: "SnowflakeCursor", stage_name: str):
    """Drop an internal stage within Snowflake.

    Args
        cursor: Snowflake connection cursor
        stage_name: Name of internal stage to drop
    """
    ev = tracing.Event("drop_internal_stage", is_parallel=False)

    drop_stage_sql = (
        f'DROP STAGE "{stage_name}" ' f"/* io.snowflake.drop_internal_stage() */ "
    )
    cursor.execute(drop_stage_sql, _is_internal=True)

    ev.finalize()


def do_upload_and_cleanup(
    cursor: "SnowflakeCursor",
    chunk_idx: int,
    chunk_path: str,
    stage_name: str,
):
    """Upload the parquet file at the given file stream or path to Snowflake
    internal stage in a parallel thread, and perform needed cleanup.

    Args
        cursor: Snowflake connection cursor
        chunk_idx: Index of the current parquet chunk
        chunk_path: Path to the file to upload
        stage_name: Snowflake internal stage name to upload files to

    Returns
        upload_thread (threading.Thread): Return the upload thread responsible
            for uploading this chunk. If False, return None. Call
            `bodo.io.helpers.join_all_threads()` on a list of threads
            generated by this function to complete the upload.
    """

    def upload_cleanup_thread_func(
        chunk_idx,
        chunk_path,
        stage_name,
    ):
        ev_upload_parquet = tracing.Event(
            f"upload_parquet_file{chunk_idx}", is_parallel=False
        )
        upload_sql = (
            f"PUT 'file://{chunk_path}' @\"{stage_name}\" AUTO_COMPRESS=FALSE "
            f"/* io.snowflake.do_upload_and_cleanup() */"
        )
        cursor.execute(upload_sql, _is_internal=True).fetchall()  # type: ignore
        ev_upload_parquet.finalize()

        # Remove chunk file
        os.remove(chunk_path)

    if SF_WRITE_OVERLAP_UPLOAD:
        th = ExceptionPropagatingThread(
            target=upload_cleanup_thread_func,
            args=(chunk_idx, chunk_path, stage_name),
        )
        th.start()
    else:
        upload_cleanup_thread_func(chunk_idx, chunk_path, stage_name)
        th = None
    return th


def create_table_handle_exists(
    cursor: "SnowflakeCursor",
    stage_name: str,
    location: str,
    sf_schema,
    if_exists: str,
    table_type: str,
):  # pragma: no cover
    """Automatically create a new table in Snowflake at the given location if
    it doesn't exist, following the schema of staged files.

    Args
        cursor: Snowflake connection cursor
        stage_name: Name of internal stage containing desired files
        location: Location to create a table
        sf_schema (dict): key: dataframe column names, value: dataframe column snowflake datatypes
        if_exists: Action to take if table already exists:
            "fail": If table exists, raise a ValueError. Create if does not exist
            "replace": If table exists, drop it, recreate it, and insert data.
                Create if does not exist
            "append": If table exists, insert data. Create if does not exist
        table_type: Type of table to create. Must be one of "", "TRANSIENT", or "TEMPORARY"

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
        raise ValueError(f"'{table_type}' is not valid value for table_type")

    if if_exists == "fail":
        create_table_cmd = f"CREATE {table_type} TABLE"
    elif if_exists == "replace":
        create_table_cmd = f"CREATE OR REPLACE {table_type} TABLE"
    elif if_exists == "append":
        create_table_cmd = f"CREATE {table_type} TABLE IF NOT EXISTS"
    else:
        raise ValueError(f"'{if_exists}' is not valid for if_exists")

    # Infer schema can return the columns out of order depending on the
    # chunking we do when we upload, so we have to iterate through the
    # dataframe columns to make sure we create the table with its columns
    # in order.
    ev_create_table = tracing.Event("create_table", is_parallel=False)

    # Snowflake requires all column start with a alphanumeric character
    # or _ to be able to write to it. Otherwise we must wrap the column in
    # quotes.
    create_table_col_lst = []
    for col_name, typ in sf_schema.items():
        col_name = (
            col_name if col_name[0].isalnum() or col_name[0] == "_" else f'"{col_name}"'
        )
        create_table_col_lst.append(f"{col_name} {typ}")
    create_table_columns = ", ".join(create_table_col_lst)
    create_table_sql = (
        f"{create_table_cmd} {location} ({create_table_columns}) "
        f"/* io.snowflake.create_table_if_not_exists() */"
    )
    cursor.execute(create_table_sql, _is_internal=True)
    ev_create_table.finalize()

    ev.finalize()


def execute_copy_into(
    cursor: "SnowflakeCursor",
    stage_name: str,
    location: str,
    sf_schema,
):  # pragma: no cover
    """Execute a COPY_INTO command from all files in stage to a table location.

    Args
        cursor: Snowflake connection cursor
        stage_name: Name of internal stage containing desired files
        location: Desired table location
        sf_schema (dict): key: dataframe column names, value: dataframe column snowflake datatypes

    Returns (nsuccess, num_chunks, num_rows, output) where:
        nsuccess (int): Number of chunks successfully copied by the function
        num_chunks (int): Number of chunks of data that the function copied
        num_rows (int): Number of rows that the function inserted
        output (str): Output of the `COPY INTO <table>` command
    """
    ev = tracing.Event("execute_copy_into", is_parallel=False)

    cols_list = []
    # Snowflake requires all column start with a alphanumeric character
    # or _ to be able to write to it. Otherwise we must wrap the column in
    # quotes.
    for col_name in sf_schema.keys():
        col_name = (
            col_name if col_name[0].isalnum() or col_name[0] == "_" else f'"{col_name}"'
        )
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

    # Time data: set as string, and data is stored correctly as time based on sf_schema info received.

    binary_time_mod = {
        c: "::binary"
        if sf_schema[c] == "BINARY"
        else "::string"
        if sf_schema[c].startswith("TIME")
        else ""
        for c in sf_schema.keys()
    }

    parquet_columns = ",".join(
        [f'$1:"{c}"{binary_time_mod[c]}' for c in sf_schema.keys()]
    )

    # Execute copy_into command with files from all ranks
    copy_into_sql = (
        f"COPY INTO {location} ({columns}) "
        f'FROM (SELECT {parquet_columns} FROM @"{stage_name}") '
        f"FILE_FORMAT=(TYPE=PARQUET COMPRESSION=AUTO BINARY_AS_TEXT=False) "
        f"PURGE=TRUE ON_ERROR={SF_WRITE_COPY_INTO_ON_ERROR} "
        f"/* io.snowflake.execute_copy_into() */"
    )
    copy_results = cursor.execute(copy_into_sql, _is_internal=True).fetchall()  # type: ignore

    # Compute diagnostic output

    # We have had instances where the output of 'fetchall' may not have tuples with expected
    # lengths or values, hence the error handling.
    def check_chunk_load_success(e) -> int:
        if isinstance(e, tuple) and (len(e) >= 2) and (e[1] == "LOADED"):
            return 1
        return 0

    def check_num_loaded_rows(e) -> int:
        if isinstance(e, tuple) and len(e) >= 4:
            try:
                return int(e[3])
            except ValueError:  # pragma: no cover
                return 0
        return 0

    nsuccess = sum(check_chunk_load_success(e) for e in copy_results)
    nchunks = len(copy_results)
    nrows = sum(check_num_loaded_rows(e) for e in copy_results)
    out = (nsuccess, nchunks, nrows, copy_results)

    ev.add_attribute("copy_into_nsuccess", nsuccess)
    ev.add_attribute("copy_into_nchunks", nchunks)
    ev.add_attribute("copy_into_nrows", nrows)

    if os.environ.get("BODO_SF_WRITE_DEBUG") is not None:
        print(f"[Snowflake Write] COPY INTO results:\n{repr(copy_results)}")
        print(
            f"[Snowflake Write] Total rows: {nrows}. Total files processed: {nchunks}. Total files successfully processed: {nsuccess}"
        )
    ev.finalize()

    return out


# ------------------- Native Distributed Snowflake Write implementation -----------------
# Register opaque type for Snowflake Cursor so it can be shared between
# different sections of jitted code.
# Skip Python type registration if snowflake.connector is not installed,
# since this is an optional dependency.
try:
    import snowflake.connector

    snowflake_connector_cursor_python_type = snowflake.connector.cursor.SnowflakeCursor
except (ImportError, AttributeError):
    snowflake_connector_cursor_python_type = None

SnowflakeConnectorCursorType = install_py_obj_class(
    types_name="snowflake_connector_cursor_type",
    python_type=snowflake_connector_cursor_python_type,
    module=sys.modules[__name__],
    class_name="SnowflakeConnectorCursorType",
    model_name="SnowflakeConnectorCursorModel",
)

# Register opaque type for TemporaryDirectory so it can be shared between
# different sections of jitted code
TemporaryDirectoryType = install_py_obj_class(
    types_name="temporary_directory_type",
    python_type=TemporaryDirectory,
    module=sys.modules[__name__],
    class_name="TemporaryDirectoryType",
    model_name="TemporaryDirectoryModel",
)


def get_snowflake_stage_info(
    cursor: "SnowflakeCursor",
    stage_name: str,
    tmp_folder: TemporaryDirectory,
) -> Dict:
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


def connect_and_get_upload_info(conn_str: str):
    """On rank 0, connect to Snowflake, create an internal stage, and issue
    an upload command to get parquet path and internal stage credentials.
    If the internal stage type is not supported or SF_WRITE_UPLOAD_USING_PUT
    is True, use the PUT implementation by connecting to Snowflake on all ranks.
    This function exists to be executed within objmode from `DataFrame.to_sql()`

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
        azure_stage_direct_upload (boolean): Whether the stage is ADLS backed
            and we'll be writing parquet files to it directly using our existing
            hdfs and parquet infrastructure.
        old_core_site (str): In case azure_stage_direct_upload=True, we replace
            bodo.HDFS_CORE_SITE_LOC with a new core-site.xml. `old_core_site`
            contains the original contents of the file (or "__none__" if file
            didn't originally exist -- see bodo.io.helpers.update_file_contents
            for more details), so that it can be restored later during
            create_table_copy_into
        old_sas_token (str): In case azure_stage_direct_upload=True, we replace
            contents in SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION (if any)
            with the SAS token for this upload. `old_sas_token`
            contains the original contents of the file (or "__none__" if file
            didn't originally exist -- see bodo.io.helpers.update_file_contents
            for more details), so that it can be restored later during
            create_table_copy_into
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
    old_core_site = ""  # Forward declaration
    sas_token = ""  # Forward declaration
    old_sas_token = ""  # Forward declaration

    err = None  # Forward declaration
    if my_rank == 0:
        try:
            # Connect to snowflake
            conn = snowflake_connect(conn_str)
            cursor = conn.cursor()
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
                    # We cannot upload directly to ADLS unless this package is installed,
                    # so check that first.
                    bodo_azurefs_sas_token_provider_installed = False
                    try:
                        import bodo_azurefs_sas_token_provider  # noqa

                        bodo_azurefs_sas_token_provider_installed = True
                    except ImportError:
                        pass

                    # Also check for environment variables such as HADOOP_HOME, ARROW_LIBHDFS_DIR
                    # and CLASSPATH, since if those are not set, the pq_write would fail anyway.
                    hadoop_env_vars_set = (
                        (len(os.environ.get("HADOOP_HOME", "")) > 0)
                        and (len(os.environ.get("ARROW_LIBHDFS_DIR", "")) > 0)
                        # CLASSPATH should be initialized by bodo_azurefs_sas_token_provider even if it
                        # didn't originally exist, but doesn't hurt to check.
                        and (len(os.environ.get("CLASSPATH", "")) > 0)
                    )

                    if (
                        bodo_azurefs_sas_token_provider_installed
                        and hadoop_env_vars_set
                    ):
                        # Upload path format: abfs[s]://<file_system>@<account_name>.dfs.core.windows.net/<path>/<file_name>
                        # E.g. abfs://stageszz05dc579c-e473-4aa2-b8a3-62a1ae425a11@qiavr8sfcb1stg.dfs.core.windows.net/<file_name>
                        # For URI syntax, see https://docs.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-introduction-abfs-uri#uri-syntax
                        container_name, _, path = upload_info["location"].partition("/")
                        path = path.rstrip("/")

                        account_name = upload_info["storageAccount"]
                        sas_token = upload_info["creds"]["AZURE_SAS_TOKEN"].lstrip("?")

                        if len(path) == 0:
                            parquet_path = f"abfs://{container_name}@{account_name}.dfs.core.windows.net/"
                        else:
                            parquet_path = f"abfs://{container_name}@{account_name}.dfs.core.windows.net/{path}/"

                        if not ("BODO_PLATFORM_WORKSPACE_UUID" in os.environ):
                            # Since setting up Hadoop can be notoriously difficult, we warn the users that
                            # they may need to fall back to the PUT method manually in case of failure.
                            # Note that this shouldn't be an issue on our platform where Hadoop is set
                            # up correctly, and is meant for on-prem users. BODO_PLATFORM_WORKSPACE_UUID
                            # should be an environment variable on all platform clusters, which is why we
                            # use this as a heuristic for showing this warning.
                            warnings.warn(
                                BodoWarning(
                                    "Detected Azure Stage. Bodo will try to upload to the stage directly. "
                                    "If this fails, there might be issues with your Hadoop configuration "
                                    "and you may need to use the PUT method instead by setting\n"
                                    "import bodo\n"
                                    "bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = True\n"
                                    "before calling this function."
                                )
                            )

                    else:
                        # If the package doesn't exist or one of the required hadoop env vars is not set,
                        # fall back to PUT, but show a warning so users can install the package
                        # and/or set up hadoop for next time.
                        fallback_to_put = True
                        warning_msg = "Detected Azure Stage. "
                        if not bodo_azurefs_sas_token_provider_installed:
                            warning_msg += (
                                "Required package bodo_azurefs_sas_token_provider "
                                "is not installed. To use direct upload to stage in the future, install the package using: "
                                "'conda install bodo-azurefs-sas-token-provider -c bodo.ai -c conda-forge'.\n"
                            )
                        if not hadoop_env_vars_set:
                            warning_msg += (
                                "You need to download and set up Hadoop. For more information, "
                                "refer to our documentation: https://docs.bodo.ai/latest/file_io/?h=hdfs#HDFS.\n"
                            )
                        warning_msg += "Falling back to PUT command for upload for now."
                        warnings.warn(BodoWarning(warning_msg))

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
            if os.environ.get("BODO_SF_WRITE_DEBUG") is not None:
                print("".join(traceback.format_exception(None, e, e.__traceback__)))

    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err

    parquet_path = comm.bcast(parquet_path)
    azure_stage_direct_upload = parquet_path.startswith("abfs://")

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

        if azure_stage_direct_upload:
            # parquet_path will be start with abfs:// if this import worked on rank-0,
            # so it should be safe to do it here on all other ranks.
            # This adds the required jars to the CLASSPATH.
            import bodo_azurefs_sas_token_provider  # noqa

            # If writing directly to an ADLS stage, we need to initialize
            # the directory for core-site and the core-site itself.
            bodo.HDFS_CORE_SITE_LOC_DIR.initialize()
            # We want to get original the contents of the files
            # (core-site.xml and sas_token.txt) so that
            # we can restore them later.
            # This happens in `create_table_copy_into`.
            old_core_site = update_file_contents(
                bodo.HDFS_CORE_SITE_LOC, SF_AZURE_WRITE_HDFS_CORE_SITE
            )
            sas_token = comm.bcast(sas_token)
            old_sas_token = update_file_contents(
                SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION, sas_token
            )

    stage_name = comm.bcast(stage_name)

    ev.finalize()
    return (
        cursor,
        tmp_folder,
        stage_name,
        parquet_path,
        upload_using_snowflake_put,
        old_creds,
        azure_stage_direct_upload,
        old_core_site,
        old_sas_token,
    )


def create_table_copy_into(
    cursor: "SnowflakeCursor",
    stage_name: str,
    location: str,
    sf_schema,
    if_exists: str,
    table_type: str,
    num_files_uploaded: int,
    old_creds,
    tmp_folder: TemporaryDirectory,
    azure_stage_direct_upload: bool,
    old_core_site: str,
    old_sas_token: str,
):
    """
    Auto-create a new table if needed, execute COPY_INTO, and clean up
    created internal stage, and restore old environment variables.
    This function exists to be executed within objmode from `DataFrame.to_sql()`

    Args
        cursor: Snowflake connection cursor
        stage_name: Name of internal stage containing files to copy_into
        location: Destination table location
        sf_schema (dict): key: dataframe column names, value: dataframe column snowflake datatypes
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
        azure_stage_direct_upload (boolean): Whether the stage is ADLS backed
            and we wrote parquet files to it directly using our existing
            hdfs and parquet infrastructure.
        old_core_site: In case azure_stage_direct_upload=True, we replaced
            bodo.HDFS_CORE_SITE_LOC with a new core-site.xml in
            connect_and_get_upload_info. `old_core_site` contains the original
            contents of the file (or "__none__" if file didn't originally
            exist -- see bodo.io.helpers.update_file_contents
            for more details), which we'll restore in this function.
        old_sas_token: In case azure_stage_direct_upload=True, we replaced
            contents of SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION with a new SAS
            token in connect_and_get_upload_info. `old_sas_token` contains the
            original contents of the file (or "__none__" if file didn't
            originally exist -- see bodo.io.helpers.update_file_contents
            for more details), which we'll restore in this function.
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
                stage_name,
                location,
                sf_schema,
                if_exists,
                table_type,
            )
            # No point of running COPY INTO if there are no files.
            if num_files_uploaded > 0:
                (nsuccess, nchunks, nrows, copy_into_result) = execute_copy_into(
                    cursor,
                    stage_name,
                    location,
                    sf_schema,
                )

                if nchunks != num_files_uploaded:
                    raise BodoError(
                        f"Snowflake write failed. Expected COPY INTO to have processed {num_files_uploaded} files, but only {nchunks} files were found. Full COPY INTO result:\n{repr(copy_into_result)}"
                    )

                if nsuccess != nchunks:
                    raise BodoError(
                        f"Snowflake write failed. {nchunks} files were loaded, but only {nsuccess} were successful. Full COPY INTO result:\n{repr(copy_into_result)}"
                    )

            commit_transaction_sql = (
                "COMMIT /* io.snowflake.create_table_copy_into() */"
            )
            cursor.execute(commit_transaction_sql)

            drop_internal_stage(cursor, stage_name)

            cursor.close()

        except Exception as e:
            err = RuntimeError(str(e))
            if os.environ.get("BODO_SF_WRITE_DEBUG") is not None:
                print("".join(traceback.format_exception(None, e, e.__traceback__)))

    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err

    # Put back the environment variables
    update_env_vars(old_creds)

    # Cleanup the folder that was created to store parquet chunks for upload
    tmp_folder.cleanup()

    # azure_stage_direct_upload will be true if direct upload to Azure was used.
    # If it was, restore the contents. It is highly unlikely
    # that there was actual contents, and in most cases the file
    # will just be removed.
    if azure_stage_direct_upload:
        update_file_contents(bodo.HDFS_CORE_SITE_LOC, old_core_site)
        update_file_contents(SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION, old_sas_token)

    ev.finalize()
